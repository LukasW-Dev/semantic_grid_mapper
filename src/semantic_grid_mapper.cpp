// semantic_grid_mapper_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_cloud.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>


using std::placeholders::_1;

class SemanticGridMapper : public rclcpp::Node {
  private:
    double log_odd_min_;
    double log_odd_max_;
    std::string input_topic_;
    std::string grid_map_topic_;
public:

  std::map<std::string, std::array<uint8_t, 3>> class_to_color_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;


  SemanticGridMapper() : Node("semantic_grid_mapper"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    this->declare_parameter("resolution", 0.1);
    this->declare_parameter("length", 20.0);
    this->declare_parameter("height", 20.0);
    this->declare_parameter("log_odd_min", -3.14);
    this->declare_parameter("log_odd_max", 3.14);
    this->declare_parameter("input_topic", "/rgb_cloud");
    this->declare_parameter("grid_map_topic", "semantic_grid_map");


    this->declare_parameter<std::string>("frame_id", "odom");
    std::string frame_id = this->get_parameter("frame_id").as_string();

    bool use_sim_time = this->get_parameter("use_sim_time").as_bool();
    this->set_parameter(rclcpp::Parameter("use_sim_time", use_sim_time));

    resolution_ = get_parameter("resolution").as_double();
    length_ = get_parameter("length").as_double();
    height_ = get_parameter("height").as_double();
    log_odd_min_ = get_parameter("log_odd_min").as_double();
    log_odd_max_ = get_parameter("log_odd_max").as_double();
    input_topic_ = get_parameter("input_topic").as_string();
    grid_map_topic_ = get_parameter("grid_map_topic").as_string();



    class_names_ = {"bush", "dirt", "fence", "grass", "gravel", "log", "mud", "object", "other-terrain",
                    "rock", "sky", "structure", "tree-foliage", "tree-trunk", "water", "unlabeled",
                    "unlabeled", "unlabeled", "unlabeled"};

    color_to_class_ = {
      {{34, 139, 34}, "bush"},
      {{139, 69, 19}, "dirt"},
      {{255, 215, 0}, "fence"},
      {{124, 252, 0}, "grass"},
      {{169, 169, 169}, "gravel"},
      {{160, 82, 45}, "log"},
      {{101, 67, 33}, "mud"},
      {{255, 0, 255}, "object"},
      {{128, 128, 0}, "other-terrain"},
      {{112, 128, 144}, "rock"},
      {{135, 206, 235}, "sky"},
      {{178, 34, 34}, "structure"},
      {{0, 100, 0}, "tree-foliage"},
      {{139, 115, 85}, "tree-trunk"},
      {{0, 191, 255}, "water"},
      {{0, 0, 0}, "unlabeled"},
      {{0, 0, 0}, "unlabeled"},
      {{0, 0, 0}, "unlabeled"},
      {{0, 0, 0}, "unlabeled"}
    };

    for (const auto& pair : color_to_class_) {
        const auto& rgb_tuple = pair.first;
        class_to_color_[pair.second] = {std::get<0>(rgb_tuple), std::get<1>(rgb_tuple), std::get<2>(rgb_tuple)};
    }

    // Probabilistic layer for each class (log odd format)
    map_ = grid_map::GridMap(class_names_);

    // Visualizatin layer for each class
    for(auto name : class_names_)
    {
      map_.add(name + "_rgb");
    }

    // Hit count layer for each class
    for(auto name : class_names_)
    {
      map_.add(name + "_hit");
    }

    // Probalility layers (% format 0..1)
    for(auto name : class_names_)
    {
      map_.add(name + "_prob");
    }

    // Visualization layer which combines all classes
    map_.add("dominant_class");

    // Initialize the map
    map_.setGeometry(grid_map::Length(length_, height_), resolution_, grid_map::Position(0.0, 0.0));
    map_.setFrameId(frame_id);

    for (const auto& name : class_names_) {
      map_[name].setConstant(0.0); // log-odds 0 => p = 0.5
    }
    map_["dominant_class"].setConstant(-1.0);

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, 0, std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));

    grid_map_pub_ = create_publisher<grid_map_msgs::msg::GridMap>(grid_map_topic_, 10);

    RCLCPP_INFO(this->get_logger(), "Semantic Grid Mapper initialized.");
  }

private:
  double prob_to_log_odds(double p) {
    return std::log(p / (1.0 - p));
  }

  double log_odds_to_prob(double l) {
    return 1.0 - (1.0 / (1.0 + std::exp(l)));
  }

  double update_log_odds(double old_l, double meas_l) {
    // If new cell (not yet initialized, only new value)
    if (std::isnan(old_l)) {
        return std::clamp(meas_l, log_odd_min_, log_odd_max_);
    }

    // Update log-odds and clamp to range [-10, 10]
    double updated_l = old_l + meas_l;
    return std::clamp(updated_l, log_odd_min_, log_odd_max_);
}

  float packRGB(uint8_t r, uint8_t g, uint8_t b) {
    uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                    static_cast<uint32_t>(g) << 8  |
                    static_cast<uint32_t>(b));
    float rgb_float;
    std::memcpy(&rgb_float, &rgb, sizeof(float));
    return rgb_float;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Move the whole map with the robot
    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer_.lookupTransform("odom", msg->header.frame_id, tf2::TimePointZero);
    
      double robot_x = transform.transform.translation.x;
      double robot_y = transform.transform.translation.y;
      //RCLCPP_INFO(this->get_logger(), "Robot Position: (%f, %f)", robot_x, robot_y);
    
      map_.move(grid_map::Position(robot_x, robot_y));
    }
    catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }


    // Create PCL Cloud
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    // Transform the points into map frame
    pcl::PointCloud<pcl::PointXYZRGB> transformed_cloud;
    Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform.transform);
    pcl::transformPointCloud(pcl_cloud, transformed_cloud, eigen_transform);

    // Reset Hit Counts
    for(auto name : class_names_)
    {
      map_[name + "_hit"].setConstant(0.0);
    }

    // Update hit count
    for (const auto& point : transformed_cloud.points) {
      std::tuple<uint8_t, uint8_t, uint8_t> color{point.r, point.g, point.b};
      if (color_to_class_.count(color) == 0) continue;
      std::string cls = color_to_class_[color];

      grid_map::Position pos(point.x, point.y);
      if (!map_.isInside(pos))
      {
        continue;
      }

      grid_map::Index idx;
      map_.getIndex(pos, idx);

      map_.at(cls + "_hit", idx) += 1.0;
    }

    // Reset probabilities
    for(auto name : class_names_)
    {
      map_[name + "_prob"].setConstant(0.0);
    }

    // Calculate class probabilities
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      grid_map::Index index = *it;
    
      float total_hits = 0.0;
    
      // Calculate the total hits across all classes at this cell
      for (const auto& class_name : class_names_) {
        if (map_.isValid(index, class_name + "_hit")) {
          total_hits += map_.at(class_name + "_hit", index);
        }
      }
    
      // Calculate normalized probabilities and store them
      for (const auto& class_name : class_names_) {
    
        float hit_count = 0.0;
        if (map_.isValid(index, class_name + "_hit")) {
          hit_count = map_.at(class_name + "_hit", index);
        }
    
        float prob = (total_hits > 0.0) ? (hit_count / total_hits) : 0.0;
        map_.at(class_name + "_prob", index) = prob;
      }
    }

    
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      grid_map::Index index = *it;

      std::array<unsigned char, 3> max_class_rgb;
      double max_log_odd = log_odd_min_;
      
      for (const auto& class_name : class_names_) {
        // Update the historical probabilities
        auto& val = map_.at(class_name, index);
        val = update_log_odds(val, prob_to_log_odds(map_.at(class_name + "_prob", index)));

        // Calculate the rgb visualizatin for the class layers
        const auto& rgb = class_to_color_[class_name];
        if(val > 0)
        {
          map_.at(class_name + "_rgb", index) = packRGB(rgb[0], rgb[1], rgb[2]);
        }
        else {
          map_.at(class_name + "_rgb", index) = std::numeric_limits<float>::quiet_NaN();;
        }

        // Store the most dominant class
        if(val > max_log_odd)
        {
          max_log_odd = val;
          max_class_rgb = rgb;
        }
      }

      // Set the dominent class rgb
      if(max_log_odd > 0)
      {
        map_.at("dominant_class", index) = packRGB(max_class_rgb[0], max_class_rgb[1], max_class_rgb[2]);
      }
    }

    grid_map_msgs::msg::GridMap map_msg;
    map_msg = *grid_map::GridMapRosConverter::toMessage(map_);
    map_msg.header.stamp = msg->header.stamp;
    grid_map_pub_->publish(map_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;

  grid_map::GridMap map_;
  std::vector<std::string> class_names_;
  std::map<std::tuple<uint8_t, uint8_t, uint8_t>, std::string> color_to_class_;

  double resolution_, length_, height_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticGridMapper>());
  rclcpp::shutdown();
  return 0;
}