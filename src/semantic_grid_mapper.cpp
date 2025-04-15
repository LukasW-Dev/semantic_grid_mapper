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
public:

  std::map<std::string, std::array<uint8_t, 3>> class_to_color_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;


  SemanticGridMapper() : Node("semantic_grid_mapper"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    this->declare_parameter("resolution", 0.1);
    this->declare_parameter("length", 20.0);
    this->declare_parameter("height", 20.0);
    this->declare_parameter<std::string>("frame_id", "odom");
    std::string frame_id = this->get_parameter("frame_id").as_string();

    bool use_sim_time = this->get_parameter("use_sim_time").as_bool();
    this->set_parameter(rclcpp::Parameter("use_sim_time", true));

    resolution_ = get_parameter("resolution").as_double();
    length_ = get_parameter("length").as_double();
    height_ = get_parameter("height").as_double();

    class_names_ = {"tree", "dirt", "fence", "grass", "gravel", "log", "mud", "object", "other-terrain",
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

    map_ = grid_map::GridMap(class_names_);
    for(auto name : class_names_)
    {
      map_.add(name + "_rgb");
    }
    map_.add("dominant_class");
    map_.setGeometry(grid_map::Length(length_, height_), resolution_, grid_map::Position(0.0, 0.0));
    map_.setFrameId(frame_id);


    for (const auto& name : class_names_) {
      map_[name].setConstant(0.0); // log-odds 0 => p = 0.5
    }
    map_["dominant_class"].setConstant(-1.0);

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rgb_cloud", 10, std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));

    grid_map_pub_ = create_publisher<grid_map_msgs::msg::GridMap>("semantic_grid_map", 10);

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
        return std::clamp(meas_l, -10.0, 10.0);
    }

    // Update log-odds and clamp to range [-10, 10]
    double updated_l = old_l + meas_l;
    return std::clamp(updated_l, -10.0, 10.0);
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

    // Process Points
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

      for (const auto& name : class_names_) {
        if (name == cls) {
          double old_log = map_.at(name, idx);
          map_.at(name, idx) = update_log_odds(old_log, prob_to_log_odds(0.7));
          const auto& rgb = class_to_color_[name];
          map_.at(name + "_rgb", idx) = packRGB(rgb[0], rgb[1], rgb[2]);
          //RCLCPP_INFO(this->get_logger(), "Updated Value: %f, Class: %s", map_.at(name, idx), name.c_str());
          //RCLCPP_INFO(this->get_logger(), "Updated Value: %f, Class: %s, Meas: %f", map_.at(name, idx), name.c_str(), prob_to_log_odds(0.7));
        }
      }
    }

    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      double max_prob = 0.0;
      int max_class = -1;

      for (size_t i = 0; i < class_names_.size(); ++i) {
        double log_val = map_.at(class_names_[i], *it);
        double prob = log_odds_to_prob(log_val);
        if (prob > max_prob) {
          max_prob = prob;
          max_class = static_cast<int>(i);
        }
      }

      //RCLCPP_INFO(this->get_logger(), "Max Class: %i", max_class);

      if (max_class >= 0) {
        // Get the RGB color from class index
        const auto& class_name = class_names_[max_class];
        const auto& rgb = class_to_color_[class_name];
    
        // Pack RGB into a float for RViz visualization
        float rgb_float = packRGB(rgb[0], rgb[1], rgb[2]);
    
        map_.at("dominant_class", *it) = rgb_float;
        //RCLCPP_INFO(this->get_logger(), "RGB: %f", rgb_float);
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