// semantic_grid_mapper_node.cpp

#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <grid_map_ros/grid_map_ros.hpp>

#include <filters/filter_chain.hpp>
#include <string>

//#include <semantic_grid_mapper/FiltersDemo.hpp>
#include "cluster.cpp"

#include <functional> // for std::hash

namespace std {
template <>
struct hash<grid_map::Index> {
  std::size_t operator()(const grid_map::Index& idx) const noexcept {
    // Combine the two ints into a hash
    return std::hash<int>()(idx.x()) ^ (std::hash<int>()(idx.y()) << 1);
  }
};

template <>
struct equal_to<grid_map::Index> {
  bool operator()(const grid_map::Index& a, const grid_map::Index& b) const noexcept {
    return a.x() == b.x() && a.y() == b.y();
  }
};
}

inline void unpackRGB(float rgb_float, uint8_t &r, uint8_t &g, uint8_t &b) {
  uint32_t rgb;
  std::memcpy(&rgb, &rgb_float, sizeof(rgb));
  r = (rgb >> 16) & 0xFF;
  g = (rgb >>  8) & 0xFF;
  b = (rgb      ) & 0xFF;
}

using std::placeholders::_1;

class SemanticGridMapper : public rclcpp::Node {
private:
  double log_odd_min_;
  double log_odd_max_;
  bool use_sim_time_;
  double resolution_;
  double length_;
  double height_;

  double robot_height_;
  double max_veg_height_;

  std::string semantic_pointcloud_topic_;
  std::string pointcloud_topic1_;
  std::string pointcloud_topic2_;
  std::string grid_map_topic_;
  std::string map_frame_id_;
  std::string robot_base_frame_id_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr semantic_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub1_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub2_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;

  grid_map::GridMap map_;
  std::vector<std::string> class_names_;
  std::map<std::tuple<uint8_t, uint8_t, uint8_t>, std::string> color_to_class_;

  std::map<std::string, std::array<uint8_t, 3>> class_to_color_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  filters::FilterChain<grid_map::GridMap> filterChain_;
  std::string filterChainParametersName_;

public:
  SemanticGridMapper()
      : Node("semantic_grid_mapper"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_), filterChain_("grid_map::GridMap") {

    // Declare all parameters expected from the YAML file
    this->declare_parameter<double>("resolution");
    this->declare_parameter<double>("length");
    this->declare_parameter<double>("height");
    this->declare_parameter<double>("log_odd_min");
    this->declare_parameter<double>("log_odd_max");

    this->declare_parameter<std::string>("semantic_pointcloud_topic");
    this->declare_parameter<std::string>("pointcloud_topic1");
    this->declare_parameter<std::string>("pointcloud_topic2");
    this->declare_parameter<std::string>("grid_map_topic");
    this->declare_parameter<std::string>("map_frame_id");
    this->declare_parameter<std::string>("robot_base_frame_id");

    this->declare_parameter<double>("robot_height");
    this->declare_parameter<double>("max_veg_height");

    this->declare_parameter("filter_chain_parameter_name", std::string("filters"));

    // Retrieve and store parameter values
    this->get_parameter("resolution", resolution_);
    this->get_parameter("length", length_);
    this->get_parameter("height", height_);
    this->get_parameter("log_odd_min", log_odd_min_);
    this->get_parameter("log_odd_max", log_odd_max_);

    this->get_parameter("robot_height", robot_height_);
    this->get_parameter("max_veg_height", max_veg_height_);

    this->get_parameter("semantic_pointcloud_topic", semantic_pointcloud_topic_);
    this->get_parameter("pointcloud_topic1", pointcloud_topic1_);
    this->get_parameter("pointcloud_topic2", pointcloud_topic2_);
    this->get_parameter("grid_map_topic", grid_map_topic_);
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("robot_base_frame_id", robot_base_frame_id_);


    this->get_parameter("use_sim_time", use_sim_time_);
    this->set_parameter(rclcpp::Parameter("use_sim_time", use_sim_time_));

    this->get_parameter("filter_chain_parameter_name", filterChainParametersName_);
    RCLCPP_INFO(this->get_logger(), "Filter chain parameter name: %s", filterChainParametersName_.c_str());

    class_names_ = {"bush",          "dirt",       "fence",    "grass",
                    "gravel",        "log",        "mud",      "object",
                    "other-terrain", "rock",       "sky",      "structure",
                    "tree-foliage",  "tree-trunk", "water",    "unlabeled",
                    "unlabeled",     "unlabeled",  "unlabeled"};

    color_to_class_ = {
        {{34, 139, 34}, "bush"},          {{139, 69, 19}, "dirt"},
        {{255, 215, 0}, "fence"},         {{124, 252, 0}, "grass"},
        {{169, 169, 169}, "gravel"},      {{160, 82, 45}, "log"},
        {{101, 67, 33}, "mud"},           {{255, 0, 255}, "object"},
        {{128, 128, 0}, "other-terrain"}, {{112, 128, 144}, "rock"},
        {{135, 206, 235}, "sky"},         {{178, 34, 34}, "structure"},
        {{0, 100, 0}, "tree-foliage"},    {{139, 115, 85}, "tree-trunk"},
        {{0, 191, 255}, "water"},         {{0, 0, 0}, "unlabeled"},
        {{0, 0, 0}, "unlabeled"},         {{0, 0, 0}, "unlabeled"},
        {{0, 0, 0}, "unlabeled"}};

    for (const auto &pair : color_to_class_) {
      const auto &rgb_tuple = pair.first;
      class_to_color_[pair.second] = {std::get<0>(rgb_tuple),
                                      std::get<1>(rgb_tuple),
                                      std::get<2>(rgb_tuple)};
    }

    // Probabilistic layer for each class (log odd format)
    map_ = grid_map::GridMap(class_names_);

    // Visualizatin layer for each class
    for (auto name : class_names_) {
      map_.add(name + "_rgb");
    }

    // Hit count layer for each class
    for (auto name : class_names_) {
      map_.add(name + "_hit");
    }

    // Probalility layers (% format 0..1)
    for (auto name : class_names_) {
      map_.add(name + "_prob");
    }

    // Probalility layers (% format 0..1)
    for (auto name : class_names_) {
      map_.add(name + "_prob");
    }

    // Independent Layer for each class. The above probabilities have cross-class dependencies. 
    // For example if a cell has a high probability to be tree-foliage, the probability for tree-trunk is low
    // even though there might be hits in the cell. This is a problem if we wan't to extract the info whether there is a tree trunk because
    // the info could get lost due to many hits for tree-foliage.
    for (auto name : class_names_) {
      map_.add(name + "_idp");
      map_[name + "_idp"].setConstant(0.0);
    }

    // Visualization layer which combines all classes
    map_.add("dominant_class");

    // Initialize the map
    map_.setGeometry(grid_map::Length(length_, height_), resolution_,
                     grid_map::Position(0.0, 0.0));
    map_.setFrameId(map_frame_id_);

    for (const auto &name : class_names_) {
      map_[name].setConstant(0.0); // log-odds 0 => p = 0.5
    }
    map_["dominant_class"].setConstant(-1.0);

    semantic_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        semantic_pointcloud_topic_, 0,
        std::bind(&SemanticGridMapper::semanticPointCloudCallback, this, _1));

    pointcloud_sub1_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic1_, 0,
        std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));

    pointcloud_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic2_, 0,
        std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));

    grid_map_pub_ =
        create_publisher<grid_map_msgs::msg::GridMap>(grid_map_topic_, 10);

    // Setup filter chain.
    if (filterChain_.configure(
        filterChainParametersName_, this->get_node_logging_interface(),
        this->get_node_parameters_interface()))
    {
      RCLCPP_INFO(this->get_logger(), "Filter chain configured.");
    } else {
      RCLCPP_ERROR(this->get_logger(), "Could not configure the filter chain!");
      rclcpp::shutdown();
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Semantic Grid Mapper initialized.");
  }

private:
  double prob_to_log_odds(double p) { return std::log(p / (1.0 - p)); }

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
                    static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    float rgb_float;
    std::memcpy(&rgb_float, &rgb, sizeof(float));
    return rgb_float;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Initialize min height layer only once
    if (!map_.exists("min_height")) {
      map_.add("min_height");
      map_.add("min_height_old");
      map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
      map_["min_height_old"].setConstant(0.0);
    }

    // Move the whole map with the robot
    geometry_msgs::msg::TransformStamped robot_transform;
    try {
      robot_transform = tf_buffer_.lookupTransform(map_frame_id_, robot_base_frame_id_, tf2::TimePointZero);

      double robot_x = robot_transform.transform.translation.x;
      double robot_y = robot_transform.transform.translation.y;
      map_.move(grid_map::Position(robot_x, robot_y));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }

    // Get transform from msg frame to map frame
    geometry_msgs::msg::TransformStamped pc_transform;
    try {
      pc_transform = tf_buffer_.lookupTransform(map_frame_id_, msg->header.frame_id,
                                               tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }

    // Create PCL Cloud
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    // remove the points which are too close to the robot
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    for (const auto &point : pcl_cloud.points) {
      if (std::hypot(point.x, point.y, point.z) >= 1.5) {
        filtered_cloud.push_back(point);
      }
      else {
        RCLCPP_DEBUG(this->get_logger(), "Point too close to robot, ignoring: (%f, %f, %f)", point.x, point.y, point.z);
      }
    }
    pcl_cloud = filtered_cloud;

    // Transform the points into map frame
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    Eigen::Affine3d eigen_transform = tf2::transformToEigen(pc_transform.transform);
    pcl::transformPointCloud(pcl_cloud, transformed_cloud, eigen_transform);

    // Point Iteration
    for (const auto &point : transformed_cloud.points) {

      // Check if the point is inside the map
      grid_map::Position pos(point.x, point.y);
      if (!map_.isInside(pos)) {
        continue;
      }

      // Ignore points which are too close (1m 3d distance)
      if (std::hypot(point.x, point.y, point.z) < 1.0) {
        continue;
      }

      // Update the min height layer
      grid_map::Index idx;
      map_.getIndex(pos, idx);
      float min_height = map_.at("min_height", idx);
      if (std::isnan(min_height) || point.z < min_height) {
        map_.at("min_height", idx) = point.z;
      }

      // Update obstacle zone layer
      if(map_.exists("min_height_filtered")) {
        min_height = map_.at("min_height_filtered", idx);
        float float_rgb = map_.at("dominant_class", idx);
        uint8_t r, g, b;
        unpackRGB(float_rgb, r, g, b);
        std::string cls = color_to_class_[std::make_tuple(r, g, b)];
        if (!std::isnan(min_height)) {
          if (cls == "tree-foliage" || cls == "tree-trunk") {           
            if (point.z > (min_height + max_veg_height_) 
            && point.z < (min_height + robot_height_)) {
              if (std::isnan(map_.at("obstacle_zone", idx))) {
                map_.at("obstacle_zone", idx) = 0.0;
              }
              map_.at("obstacle_zone", idx) += 1.0;
              // RCLCPP_INFO(this->get_logger(), "obstacle zone: %f", point.z);
            }
          }
        }
      }
    }
  }

  void semanticPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Move the whole map with the robot
    geometry_msgs::msg::TransformStamped robot_transform;
    try {
      robot_transform = tf_buffer_.lookupTransform(map_frame_id_, robot_base_frame_id_, tf2::TimePointZero);

      double robot_x = robot_transform.transform.translation.x;
      double robot_y = robot_transform.transform.translation.y;
      map_.move(grid_map::Position(robot_x, robot_y));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }

    // Get transform from msg frame to map frame
    geometry_msgs::msg::TransformStamped pc_transform;
    try {
      pc_transform = tf_buffer_.lookupTransform(map_frame_id_, msg->header.frame_id, msg->header.stamp);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }

    // Initialize min height layer only once
    if (!map_.exists("min_height")) {
      map_.add("min_height");
      map_.add("min_height_old");
      map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
      map_["min_height_old"].setConstant(std::numeric_limits<float>::quiet_NaN());
    }

    // Initialize vegetation zone layer only once
    if (!map_.exists("vegetation_zone")) {
      map_.add("vegetation_zone");
      map_["vegetation_zone"].setConstant(std::numeric_limits<float>::quiet_NaN());
    }

    // Initialize obstacle zone layer only once
    if (!map_.exists("obstacle_zone")) {
      map_.add("obstacle_zone");
    }
    


    // Create PCL Cloud
    pcl::PointCloud<pcl::PointXYZRGBL> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    // Transform the points into map frame
    pcl::PointCloud<pcl::PointXYZRGBL> transformed_cloud;
    Eigen::Affine3d eigen_transform =
        tf2::transformToEigen(pc_transform.transform);
    pcl::transformPointCloud(pcl_cloud, transformed_cloud, eigen_transform);

    // Reset Hit Counts
    for (auto name : class_names_) {
      map_[name + "_hit"].setConstant(0.0);
    }

    // Point Iteration
    for (const auto &point : transformed_cloud.points) {
      std::tuple<uint8_t, uint8_t, uint8_t> color{point.r, point.g, point.b};
      
      // Reject points with no color (should actually not happen)
      if (color_to_class_.count(color) == 0)
        continue;

      // Reject points with low confidence (TODO: should this be done already in fusion?)
      if (point.label < 70 ) {
        continue;
      }
      
      // Check if the point is inside the map
      grid_map::Position pos(point.x, point.y);
      if (!map_.isInside(pos)) {
        continue;
      }

      // Update the min height layer
      grid_map::Index idx;
      map_.getIndex(pos, idx);
      float min_height = map_.at("min_height", idx);
      if (std::isnan(min_height) || point.z < min_height) {
        map_.at("min_height", idx) = point.z;
      }
      
      // Update class hit count
      std::string cls = color_to_class_[color];
      map_.at(cls + "_hit", idx) += 1.0;

      
      // Update obstacle zone layer
      if (!std::isnan(min_height)) {
        if (cls == "tree-foliage" || cls == "tree-trunk") {
          //RCLCPP_INFO(this->get_logger(), "z %f, min_height %f, max_veg_height %f, robot_height %f", point.z, min_height, max_veg_height_, robot_height_);

          if (point.z > (min_height + max_veg_height_) 
          && point.z < (min_height + robot_height_)) {
            if (std::isnan(map_.at("obstacle_zone", idx))) {
              map_.at("obstacle_zone", idx) = 0.0;
            }
            map_.at("obstacle_zone", idx) += 1.0;
            // RCLCPP_INFO(this->get_logger(), "obstacle zone: %f", point.z);
          }
        }
      }
    }

    // Reset probabilities
    for (auto name : class_names_) {
      map_[name + "_prob"].setConstant(0.0);
    }

    // Map Iteration 1
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      grid_map::Index index = *it;

      // Calculate the total hits across all classes at this cell
      float total_hits = 0.0;
      for (const auto &class_name : class_names_) {
        if (map_.isValid(index, class_name + "_hit")) {
          total_hits += map_.at(class_name + "_hit", index);
        }
      }

      // needed for log odd calculation
      std::array<unsigned char, 3> max_class_rgb;
      double max_log_odd = log_odd_min_;

      // Calculate normalized probabilities and store them
      for (const auto &class_name : class_names_) {
        float hit_count = 0.0;
        if (map_.isValid(index, class_name + "_hit")) {
          hit_count = map_.at(class_name + "_hit", index);
        }
        float prob = (total_hits > 0.0) ? (hit_count / total_hits) : 0.0;
        map_.at(class_name + "_prob", index) = prob;

        // Decrease the indepentend representation for each class each update a bit for clearance
        // TODO: Find better way to do this
        {
          auto& val = map_.at(class_name + "_idp", index);
          val -= 0.3;
          if(val < -1)
          {
            val = -1;
          }
          float hit_count = 0.0;
          if (map_.isValid(index, class_name + "_hit")) {
            hit_count = map_.at(class_name + "_hit", index);
          }
          if (std::isnan(val)) {
            val = 0.0;
          }
          val += hit_count;

          // Calculate the rgb visualizatin for the class layers
          const auto &rgb = class_to_color_[class_name];
          if (val > 0) {
            map_.at(class_name + "_rgb", index) = packRGB(rgb[0], rgb[1], rgb[2]);
          } else {
            map_.at(class_name + "_rgb", index) = std::numeric_limits<float>::quiet_NaN();
          }
        }

        // Update the historical probabilities
        {
          auto &val = map_.at(class_name, index);
          val = update_log_odds(
              val, prob_to_log_odds(map_.at(class_name + "_prob", index)));

          // Calculate the rgb visualizatin for the class layers
          const auto &rgb = class_to_color_[class_name];

          // Store the most dominant class
          if (val > max_log_odd) {
            max_log_odd = val;
            max_class_rgb = rgb;
          }
        }
      }

      // Set obstacle zone to nan if too few (noise)
      if (map_.at("obstacle_zone", index) < 5.0) {
        map_.at("obstacle_zone", index) = std::numeric_limits<float>::quiet_NaN();
      }

      
      // Set the dominent class rgb
      if (max_log_odd > 0) {
        map_.at("dominant_class", index) =
        packRGB(max_class_rgb[0], max_class_rgb[1], max_class_rgb[2]);
      }

      // If min height is NaN, use value from the old layer
      if (std::isnan(map_.at("min_height", index))) {
        map_.at("min_height", index) = map_.at("min_height_old", index);
      }
    }

    // Fill the obstacle zone
    std::string obstacle_zone = "obstacle_zone";

    // Print amount of points in the obstacle zone
    int count = 0;
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      grid_map::Index index = *it;
      if (map_.at(obstacle_zone, index) > 0.0) {
        count++;
      }
    }

    markAlphaShapeObstacleClusters(map_, obstacle_zone, 2);
    count = 0;
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      grid_map::Index index = *it;
      if (map_.at(obstacle_zone, index) > 0.0) {
        count++;
      }
    }

    // Apply filter chain.
    grid_map::GridMap min_height_filtered;
    if (!filterChain_.update(map_, min_height_filtered)) {
      RCLCPP_ERROR(this->get_logger(), "Could not update the grid map filter chain!");
      return;
    }

    grid_map_msgs::msg::GridMap map_msg;
    map_msg = *grid_map::GridMapRosConverter::toMessage(min_height_filtered);
    map_msg.header.stamp = msg->header.stamp;
    grid_map_pub_->publish(map_msg);
    map_["min_height_old"] = map_["min_height"];
    map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["obstacle_zone"].setConstant(std::numeric_limits<float>::quiet_NaN());
  }

};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticGridMapper>());
  rclcpp::shutdown();
  return 0;
}