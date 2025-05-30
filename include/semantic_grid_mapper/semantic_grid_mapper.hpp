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

#include <functional> // for std::hash

#include <chrono>

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

double prob_to_log_odds(double p) { return std::log(p / (1.0 - p)); }

double log_odds_to_prob(double l) { return 1.0 - (1.0 / (1.0 + std::exp(l))); }

float packRGB(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                  static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  float rgb_float;
  std::memcpy(&rgb_float, &rgb, sizeof(float));
  return rgb_float;
}

struct ClassLayers {
  grid_map::Matrix* hit_ground;
  grid_map::Matrix* hit_obstacle;
  //grid_map::Matrix* hit_sky;
  grid_map::Matrix* prob_ground;
  grid_map::Matrix* prob_obstacle;
  //grid_map::Matrix* prob_sky;
  grid_map::Matrix* hist_ground;
  grid_map::Matrix* hist_obstacle;
  //grid_map::Matrix* hist_sky;
  std::array<unsigned char,3> rgb;
};

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

  std::vector<ClassLayers> cls_;

  builtin_interfaces::msg::Time last_cloud_stamp_;

  rclcpp::TimerBase::SharedPtr update_timer_;

public:
  SemanticGridMapper();

  ~SemanticGridMapper() = default;

private:

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  void semanticPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  double update_log_odds(double old_l, double meas_l);

  void updateMap();

};