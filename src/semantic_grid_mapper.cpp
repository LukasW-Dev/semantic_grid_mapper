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

#include "morphology.cpp"
#include "cluster.cpp"

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
  float log_odd_min_;
  float log_odd_max_;
  bool use_sim_time_;
  double resolution_;
  double length_;
  double height_;

  double robot_height_;
  double max_veg_height_;
  double max_sky_height_;

  std::string semantic_pointcloud_topic_;
  std::string pointcloud_topic1_;
  std::string pointcloud_topic2_;
  std::string pointcloud_topic3_;
  std::string grid_map_topic_;
  std::string map_frame_id_;
  std::string robot_base_frame_id_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr semantic_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub1_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub2_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub3_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;

  grid_map::GridMap map_;
  std::vector<std::string> class_names_;
  std::map<std::tuple<uint8_t, uint8_t, uint8_t>, std::string> color_to_class_;

  std::map<std::string, std::array<uint8_t, 3>> class_to_color_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  filters::FilterChain<grid_map::GridMap> filterChain_;
  std::string filterChainParametersName_;

  //std::vector<ClassLayers> cls_;
  std::unordered_map<std::string, ClassLayers> cls_;
  rclcpp::TimerBase::SharedPtr update_timer_;

  grid_map::Matrix* min_height_;
  grid_map::Matrix* min_height_old_;
  grid_map::Matrix* min_height_smooth_;

  grid_map::Matrix* ground_class_;
  grid_map::Matrix* obstacle_class_;

  grid_map::Matrix* obstacle_zone_;
  grid_map::Matrix* obstacle_hit_count_;
  grid_map::Matrix* obstacle_;

  grid_map::Matrix* sky_zone_;
  grid_map::Matrix* sky_hit_count_;
  grid_map::Matrix* sky_;

  rclcpp::Time last_update_stamp_;

  int pc_updates_;

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
    this->declare_parameter<std::string>("pointcloud_topic3");
    this->declare_parameter<std::string>("grid_map_topic");
    this->declare_parameter<std::string>("map_frame_id");
    this->declare_parameter<std::string>("robot_base_frame_id");

    this->declare_parameter<double>("robot_height");
    this->declare_parameter<double>("max_veg_height");
    this->declare_parameter<double>("max_sky_height");

    this->declare_parameter("filter_chain_parameter_name", std::string("filters"));

    // Retrieve and store parameter values
    this->get_parameter("resolution", resolution_);
    this->get_parameter("length", length_);
    this->get_parameter("height", height_);
    this->get_parameter("log_odd_min", log_odd_min_);
    this->get_parameter("log_odd_max", log_odd_max_);

    this->get_parameter("robot_height", robot_height_);
    this->get_parameter("max_veg_height", max_veg_height_);
    this->get_parameter("max_sky_height", max_sky_height_);


    this->get_parameter("semantic_pointcloud_topic", semantic_pointcloud_topic_);
    this->get_parameter("pointcloud_topic1", pointcloud_topic1_);
    this->get_parameter("pointcloud_topic2", pointcloud_topic2_);
    this->get_parameter("pointcloud_topic3", pointcloud_topic3_);
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

    // Probabilistic layer for each class (log odd format) and each area (ground, obstacle, sky)
    for (auto name : class_names_) {
      map_.add(name + "_ground");
      map_.add(name + "_obstacle");
    }

    // Visualization layer for each class
    // for (auto name : class_names_) {
    //   map_.add(name + "_rgb");
    // }

    // Hit count layer for each class
    for (auto name : class_names_) {
      map_.add(name + "_ground_hit");
      map_.add(name + "_obstacle_hit");
    }

    // Probability layers (% format 0..1)
    for (auto name : class_names_) {
      map_.add(name + "_ground_prob");
      map_.add(name + "_obstacle_prob");
    }

    // Visualization layer which combines all classes
    map_.add("ground_class");
    map_["ground_class"].setConstant(-1.0);
    ground_class_ = &map_["ground_class"];

    map_.add("obstacle_class");
    map_["obstacle_class"].setConstant(-1.0);
    obstacle_class_ = &map_["obstacle_class"];

    map_.add("obstacle_zone");
    map_["obstacle_zone"].setConstant(0.0); 
    obstacle_zone_ = &map_["obstacle_zone"];

    map_.add("obstacle_hit_count");
    map_["obstacle_hit_count"].setConstant(0.0);
    obstacle_hit_count_ = &map_["obstacle_hit_count"];

    map_.add("obstacle"); 
    map_["obstacle"].setConstant(0.0);
    obstacle_ = &map_["obstacle"];

    map_.add("sky_zone");
    map_["sky_zone"].setConstant(0.0); 
    sky_zone_ = &map_["sky_zone"];

    map_.add("sky_hit_count");
    map_["sky_hit_count"].setConstant(0.0);
    sky_hit_count_ = &map_["sky_hit_count"];

    map_.add("sky_map"); 
    map_["sky_map"].setConstant(0.0);
    sky_ = &map_["sky_map"];
  
    map_.add("min_height");
    map_.add("min_height_old");
    map_.add("min_height_smooth");
    map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["min_height_old"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["min_height_smooth"].setConstant(std::numeric_limits<float>::quiet_NaN());
    min_height_ = &map_["min_height"];
    min_height_old_ = &map_["min_height_old"];
    min_height_smooth_ = &map_["min_height_smooth"];


    // Initialize the map
    map_.setGeometry(grid_map::Length(length_, height_), resolution_,
                     grid_map::Position(0.0, 0.0));
    map_.setFrameId(map_frame_id_);

    for (const auto &name : class_names_) {
      map_[name + "_ground"].setConstant(0.0); // log-odds 0 => p = 0.5
      map_[name + "_obstacle"].setConstant(0.0); // log-odds 0 => p = 0.5
      // map_[name + "_sky"].setConstant(0.0); // log-odds 0 => p = 0.5
    }

    rclcpp::QoS qos_profile(rclcpp::KeepLast(1));
    qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);

    semantic_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        semantic_pointcloud_topic_, qos_profile,
        std::bind(&SemanticGridMapper::semanticPointCloudCallback, this, _1));

    pointcloud_sub1_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic1_, qos_profile,
        std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));

    if(pointcloud_topic2_ != "None")
    {
      pointcloud_sub2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          pointcloud_topic2_, qos_profile,
          std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));
    }

    if(pointcloud_topic3_ != "None")
    {
      pointcloud_sub3_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          pointcloud_topic3_, qos_profile,
          std::bind(&SemanticGridMapper::pointCloudCallback, this, _1));
    }

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

    for (auto const & name : class_names_) {
      cls_[name] = {
        &map_.get(name + "_ground_hit"),
        &map_.get(name + "_obstacle_hit"),
        //&map_.get(name + "_sky_hit"),
        &map_.get(name + "_ground_prob"),
        &map_.get(name + "_obstacle_prob"),
        //&map_.get(name + "_sky_prob"),
        &map_.get(name + "_ground"),
        &map_.get(name + "_obstacle"),
        //&map_.get(name + "_sky"),
        class_to_color_[name]
      };
    }

    // Initialize Timer with 500ms period
    update_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&SemanticGridMapper::applyFilterChain, this));

    pc_updates_ = 0;

    RCLCPP_INFO(this->get_logger(), "Semantic Grid Mapper initialized.");
  }

private:
  double prob_to_log_odds(double p) {
    p = std::clamp(p, 1e-6, 1.0 - 1e-6);
    return std::log(p / (1.0 - p));
  }

  double update_log_odds(double old_l, double meas_l) {
    // If new cell (not yet initialized, only new value)
    if (std::isnan(old_l)) {
      return meas_l;
    }


    // Update log-odds and clamp to range 
    double updated_l = old_l + meas_l;
    return updated_l;
  }

  float packRGB(uint8_t r, uint8_t g, uint8_t b) {
    uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                    static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    float rgb_float;
    std::memcpy(&rgb_float, &rgb, sizeof(float));
    return rgb_float;
  }

  grid_map::Polygon createAnnularSectorPolygon(double inner_radius, double outer_radius,
                                              double min_angle, double max_angle,
                                              int num_points = 30)
  {
    grid_map::Polygon polygon;
    grid_map::Position center = map_.getPosition();  // robot is at map center

    // Handle angle wrapping
    double angle_span = max_angle >= min_angle ? max_angle - min_angle : 2.0 * M_PI - (min_angle - max_angle);

    // Outer arc (counter-clockwise)
    for (int i = 0; i <= num_points; ++i) {
        double angle = wrapTo2Pi(min_angle + i * angle_span / num_points);
        double x = center.x() + outer_radius * std::cos(angle);
        double y = center.y() + outer_radius * std::sin(angle);
        polygon.addVertex(grid_map::Position(x, y));
    }

    // Inner arc (clockwise — reverse order)
    for (int i = num_points; i >= 0; --i) {
        double angle = wrapTo2Pi(min_angle + i * angle_span / num_points);
        double x = center.x() + inner_radius * std::cos(angle);
        double y = center.y() + inner_radius * std::sin(angle);
        polygon.addVertex(grid_map::Position(x, y));
    }

    return polygon;
  }

  double wrapTo2Pi(double angle_rad)
  {
      double two_pi = 2.0 * M_PI;
      return std::fmod(std::fmod(angle_rad, two_pi) + two_pi, two_pi);
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Get Timestamp using chrono
    auto timestamp = std::chrono::steady_clock::now();
    pc_updates_++;
    map_["obstacle_hit_count"].setConstant(0.0);
    map_["sky_hit_count"].setConstant(0.0);

    //=================================================================================================================
    // 1) Move the whole map with the robot
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

    //=================================================================================================================
    // 2) Transform the points from sensor frame into robot frame
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    geometry_msgs::msg::TransformStamped pc_transform;
    try {
      pc_transform = tf_buffer_.lookupTransform(robot_base_frame_id_, msg->header.frame_id, tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    Eigen::Affine3d eigen_transform = tf2::transformToEigen(pc_transform.transform);
    pcl::transformPointCloud(pcl_cloud, transformed_cloud, eigen_transform);

    //=================================================================================================================
    // 3) Filter points
    // identify lidar
    bool top_ouster = msg->header.frame_id == "top_os_lidar";
    bool front_ouster = msg->header.frame_id == "front_os_lidar";
    bool front_livox = msg->header.frame_id == "front_livox_link";
    bool left_laser_pandar = msg->header.frame_id == "left_laser_mount";
    bool right_laser_pandar = msg->header.frame_id == "right_laser_mount";

    pcl::PointCloud<pcl::PointXYZ> obstacle_cloud;
    for (const auto &point : transformed_cloud.points) {

      // Ignore points over 1.5m
      if(point.z > 1.5){
        continue;
      }

      // Points too close to the robot are ignored
      double min_radius;
      if(left_laser_pandar || right_laser_pandar) min_radius = 1.5;
      if(top_ouster || front_ouster || front_livox) min_radius = 3.2;
      if (!(std::hypot(point.x, point.y, point.z) >= min_radius)) {
        continue;
      }

      // ========== Top Ouster ==========
      // filter points in angle +- 20 degrees
      if (top_ouster) {
        double angle = std::atan2(point.y, point.x);
        if (angle > -0.3491 && angle < 0.3491) { // -20 to +20 degrees
          continue;
        }
      }

      // filter points too close to the robot, since top ouster does not reach the ground within 3.5m radius
      if (top_ouster && std::hypot(point.x, point.y, point.z) < 3.5) {
        continue;
      }

      // ========== Front Ouster ==========
      // filter points in negative x direction
      if (front_ouster && point.x < 1) {
        continue;
      }

      // ========== Front Livox ==========
      // filter points in negative x direction
      if (front_livox && point.x < 1) {
        continue;
      }

      // ========== Left Laser Pandar / Right Laser Pandar ==========
      // filter points in negative x direction
      if((left_laser_pandar || right_laser_pandar) && point.x < 1)
      {
        continue;
      }

      obstacle_cloud.push_back(point);
    }

    //=================================================================================================================
    // 4) Transform from the robot frame to the map frame
    geometry_msgs::msg::TransformStamped map_transform;
    try {
      map_transform = tf_buffer_.lookupTransform(map_frame_id_, robot_base_frame_id_, tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }
    pcl::PointCloud<pcl::PointXYZ> map_cloud_obstacle;
    pcl::PointCloud<pcl::PointXYZ> map_cloud_sky;
    Eigen::Affine3d robot_eigen_transform = tf2::transformToEigen(map_transform.transform);
    pcl::transformPointCloud(obstacle_cloud, map_cloud_obstacle, robot_eigen_transform);
    pcl::transformPointCloud(transformed_cloud, map_cloud_sky, robot_eigen_transform);
   
    //=================================================================================================================
    // 5) Point Iteration Obstacle / Min Height
    for (const auto &point : map_cloud_obstacle.points) {

      // Check if the point is inside the map
      grid_map::Position pos(point.x, point.y);
      if (!map_.isInside(pos)) {
        continue;
      }

      // Update the min height layer
      grid_map::Index idx;
      map_.getIndex(pos, idx);
      float min_height_val = (*min_height_)(idx(0), idx(1));
      if (std::isnan(min_height_val) || point.z < min_height_val) {
        (*min_height_)(idx(0), idx(1)) = point.z;
      }

      // Update obstacle hit count
      if(point.z > (*min_height_)(idx(0), idx(1)) + max_veg_height_ && 
         point.z < (*min_height_)(idx(0), idx(1)) + robot_height_) {
        
        // Check if the point is a obstacle class
        if(std::isnan((*obstacle_class_)(idx(0), idx(1)))) continue;
        std::tuple<uint8_t, uint8_t, uint8_t> color;
        unpackRGB((*obstacle_class_)(idx(0), idx(1)), std::get<0>(color), std::get<1>(color), std::get<2>(color));
        std::string cls_name = color_to_class_[color];
        if(cls_name != "grass" && cls_name != "dirt" && cls_name != "gravel" && cls_name != "mud" && cls_name != "water")
        {
          (*obstacle_hit_count_)(idx(0), idx(1))++;
        }
      }
    }

    //=================================================================================================================
    // 6) Point Iteration Sky
    for (const auto &point : map_cloud_sky.points) {

      // Check if the point is inside the map
      grid_map::Position pos(point.x, point.y);
      if (!map_.isInside(pos)) {
        continue;
      }

      grid_map::Index idx;
      map_.getIndex(pos, idx);

      // Update sky hit count
      if(point.z > (*min_height_)(idx(0), idx(1)) + robot_height_ && 
         point.z < (*min_height_)(idx(0), idx(1)) + max_sky_height_) {
        (*sky_hit_count_)(idx(0), idx(1))++;
      }
    }

    //=================================================================================================================
    // 6) Map Iteration
    double roll, pitch, yaw;
    tf2::Quaternion tf_q;
    tf2::fromMsg(robot_transform.transform.rotation, tf_q);
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    double clipped_range = 10.0; // approximatelly half the diagonal
    
    // ========== Top Ouster ==========
    if(top_ouster)
    {
      double angle_min = wrapTo2Pi(20.0 * (M_PI / 180.0) + yaw);
      double angle_max = wrapTo2Pi(-20.0 * (M_PI / 180.0) + yaw);
      grid_map::Polygon sector = createAnnularSectorPolygon(3.5, clipped_range, angle_min, angle_max);

      for (grid_map::PolygonIterator it(map_, sector); !it.isPastEnd(); ++it) {
        grid_map::Index idx = *it;
        
        // Process each cell inside the sector
        double prob = (*obstacle_hit_count_)(idx(0), idx(1)) > 0 ? 1.0 : 0.2;
        (*obstacle_zone_)(idx(0), idx(1)) = update_log_odds((*obstacle_zone_)(idx(0), idx(1)), prob_to_log_odds(prob));
      }
    }

    // ========== Front Ouster ==========
    else if(front_ouster)
    {
      double angle_min = wrapTo2Pi(-80.0 * (M_PI / 180.0) + yaw);
      double angle_max = wrapTo2Pi(80.0 * (M_PI / 180.0) + yaw);

      grid_map::Polygon sector = createAnnularSectorPolygon(1.5, clipped_range, angle_min, angle_max);

      for (grid_map::PolygonIterator it(map_, sector); !it.isPastEnd(); ++it) {
        grid_map::Index idx = *it;
        
        // Process each cell inside the sector
        double prob = (*obstacle_hit_count_)(idx(0), idx(1)) > 0 ? 1.0 : 0.2;
        (*obstacle_zone_)(idx(0), idx(1)) = update_log_odds((*obstacle_zone_)(idx(0), idx(1)), prob_to_log_odds(prob));
        (*obstacle_zone_)(idx(0), idx(1)) = std::clamp((*obstacle_zone_)(idx(0), idx(1)), log_odd_min_, log_odd_max_);
      }
    }

    // ========== Front Livox ==========
    else if(front_livox)
    {
      double angle_min = wrapTo2Pi(-80.0 * (M_PI / 180.0) + yaw);
      double angle_max = wrapTo2Pi(80.0 * (M_PI / 180.0) + yaw);

      grid_map::Polygon sector = createAnnularSectorPolygon(1.5, clipped_range, angle_min, angle_max);

      for (grid_map::PolygonIterator it(map_, sector); !it.isPastEnd(); ++it) {
        grid_map::Index idx = *it;
        
        // Process each cell inside the sector
        double prob = (*obstacle_hit_count_)(idx(0), idx(1)) > 0 ? 1.0 : 0.2;
        (*obstacle_zone_)(idx(0), idx(1)) = update_log_odds((*obstacle_zone_)(idx(0), idx(1)), prob_to_log_odds(prob));
        (*obstacle_zone_)(idx(0), idx(1)) = std::clamp((*obstacle_zone_)(idx(0), idx(1)), log_odd_min_, log_odd_max_);
      }
    }

    // ========== Left Laser Pandar / Right Laser Pandar ==========
    else if(left_laser_pandar || right_laser_pandar)
    {
      double angle_min = wrapTo2Pi(-90.0 * (M_PI / 180.0) + yaw);
      double angle_max = wrapTo2Pi(90.0 * (M_PI / 180.0) + yaw);

      grid_map::Polygon sector = createAnnularSectorPolygon(1.5, clipped_range, angle_min, angle_max);

      for (grid_map::PolygonIterator it(map_, sector); !it.isPastEnd(); ++it) {
        grid_map::Index idx = *it;
        
        // Process each cell inside the sector
        double prob = (*obstacle_hit_count_)(idx(0), idx(1)) > 0 ? 1.0 : 0.2;
        (*obstacle_zone_)(idx(0), idx(1)) = update_log_odds((*obstacle_zone_)(idx(0), idx(1)), prob_to_log_odds(prob));
        (*obstacle_zone_)(idx(0), idx(1)) = std::clamp((*obstacle_zone_)(idx(0), idx(1)), log_odd_min_, log_odd_max_);
      }
    }

    // Map Iteration Sky Layer
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      const size_t idx = it.getLinearIndex();

      double prob = (*sky_hit_count_)(idx) > 0 ? 1.0 : 0.48;
      (*sky_zone_)(idx) = update_log_odds((*sky_zone_)(idx), prob_to_log_odds(prob));
      (*sky_zone_)(idx) = std::clamp((*sky_zone_)(idx), log_odd_min_, log_odd_max_);
    }


    last_update_stamp_ = msg->header.stamp;
    auto pc_cb_time = std::chrono::steady_clock::now();
    // RCLCPP_INFO(this->get_logger(), "PointCloud Callback took %f ms", std::chrono::duration<double, std::milli>(pc_cb_time - timestamp).count());
  }

  void semanticPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Get Timestamp using chrono
    auto timestamp = std::chrono::steady_clock::now();

    /*****************************************************************************************************************/
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

    // Measure Robot Transform Time
    auto robot_transform_time = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(this->get_logger(), "Robot transform took %f ms", std::chrono::duration<double, std::milli>(robot_transform_time - timestamp).count());

    /*****************************************************************************************************************/
    // Transform the points into map frame
    geometry_msgs::msg::TransformStamped pc_transform;
    try {
      pc_transform = tf_buffer_.lookupTransform(map_frame_id_, msg->header.frame_id, msg->header.stamp);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
      return;
    }
    pcl::PointCloud<pcl::PointXYZRGBL> transformed_cloud;
    Eigen::Affine3d eigen_transform = tf2::transformToEigen(pc_transform.transform);
    pcl::PointCloud<pcl::PointXYZRGBL> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    pcl::transformPointCloud(pcl_cloud, transformed_cloud, eigen_transform);

    // Measure Point Cloud Transform Time
    auto pointcloud_transform_time = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(this->get_logger(), "Point cloud transform took %f ms", std::chrono::duration<double, std::milli>(pointcloud_transform_time - robot_transform_time).count());

    
    // Reset Hit Counts
    for (auto& [name, cls] : cls_) {
      cls.hit_ground->setConstant(0.0);
      cls.hit_obstacle->setConstant(0.0);
    }

    // Measure Layer Initialization Time
    auto layer_initialization_time = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(this->get_logger(), "Layer initialization took %f ms", std::chrono::duration<double, std::milli>(layer_initialization_time - pointcloud_transform_time).count());

    // Vector to store visited map indices
    std::unordered_set<grid_map::Index> visited_indices;

    // Point Iteration
    for (const auto &point : transformed_cloud.points) 
    {
      std::tuple<uint8_t, uint8_t, uint8_t> color{point.r, point.g, point.b};

      // Reject points with no color (should actually not happen)
      if (color_to_class_.count(color) == 0)
        continue;

      // Reject points with low confidence (TODO: should this be done already in fusion?)
      if (point.label < 90 ) {
        continue;
      }
      
      // Check if the point is inside the map
      grid_map::Position pos(point.x, point.y);
      if (!map_.isInside(pos)) {
        continue;
      }

      grid_map::Index idx;
      map_.getIndex(pos, idx);
      visited_indices.insert(idx);
      
      // Update class hit count
      std::string cls_name = color_to_class_[color];
      
      if(map_.exists("min_height_smooth")) {

        // Ground Layer
        if(point.z < (*min_height_smooth_)(idx(0), idx(1)) + max_veg_height_) {
          (*(cls_[cls_name].hit_ground))(idx(0), idx(1)) += 1.0;
        }

        // Obstacle Layer
        else if(point.z >= (*min_height_smooth_)(idx(0), idx(1)) + max_veg_height_ 
        && point.z < (*min_height_smooth_)(idx(0), idx(1)) + robot_height_) {
          // only add hit if class is not grass or dirt or gravel
          if (cls_name != "grass" && cls_name != "dirt" && cls_name != "gravel" && cls_name != "mud" && cls_name != "water") {
            (*(cls_[cls_name].hit_obstacle))(idx(0), idx(1)) += 1.0;
          }
        }
      }

    }

    // Measure Point Iteration Time
    auto point_iteration_time = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(this->get_logger(), "Point iteration took %f ms", std::chrono::duration<double, std::milli>(point_iteration_time - layer_initialization_time).count());

    // Reset probabilities
    for (auto& [name, cls] : cls_) {
      cls.prob_ground->setConstant(0.0);
      cls.prob_obstacle->setConstant(0.0);
      // cls.prob_sky->setConstant(0.0);
    }

    // Map Iteration 1 (only for the cells that have been visited)
    for (const auto &idx : visited_indices) {
      const std::size_t i = idx(0) + idx(1) * map_.getSize()(0);

      // Calculate the total hits across all classes at this cell
      float total_ground_hits = 0.0;
      float total_obstacle_hits = 0.0;
      // float total_sky_hits = 0.0;

      for (auto& [name, cls] : cls_) {
        // Ground Layer
        total_ground_hits += (*(cls.hit_ground))(i);
        // Obstacle Layer
        total_obstacle_hits += (*(cls.hit_obstacle))(i);
        // // Sky Layer
        // total_sky_hits += cls.hit_sky->at(index);
      }

      // needed for log odd calculation
      std::array<unsigned char, 3> max_class_rgb_ground;
      double max_log_odd_ground = log_odd_min_;
      std::array<unsigned char, 3> max_class_rgb_obstacle;
      double max_log_odd_obstacle = log_odd_min_;
      // std::array<unsigned char, 3> max_class_rgb_sky;
      // double max_log_odd_sky = log_odd_min_;

      // Calculate normalized probabilities and store them
      for (auto& [name, cls] : cls_) {
        
        // Ground Layer
        float prob_ground = total_ground_hits ? (*(cls.hit_ground))(i) / total_ground_hits : 0.0;
        (*(cls.prob_ground))(i) = prob_ground;

        // Obstacle Layer
        float prob_obstacle = total_obstacle_hits ? (*(cls.hit_obstacle))(i) / total_obstacle_hits : 0.0;
        (*(cls.prob_obstacle))(i) = prob_obstacle;

        // // Sky Layer
        // (*(cls.prob_sky))(index(0), index(1)) = total_sky_hits ? cls.hit_sky->at(index) / total_sky_hits : 0.0;

        // Print hist_ground before and after update
        double log_odds_ground = update_log_odds(
            (*(cls.hist_ground))(i), prob_to_log_odds(prob_ground));

        double log_odds_obstacle = update_log_odds(
            (*(cls.hist_obstacle))(i), prob_to_log_odds(prob_obstacle));

        (*(cls.hist_ground))(i) = log_odds_ground;
        (*(cls.hist_obstacle))(i) = log_odds_obstacle;

        if(log_odds_ground > max_log_odd_ground) { 
          max_log_odd_ground = log_odds_ground; 
          max_class_rgb_ground = cls.rgb; 
        }
        if(log_odds_obstacle > max_log_odd_obstacle) { 
          max_log_odd_obstacle = log_odds_obstacle; 
          max_class_rgb_obstacle = cls.rgb; 
        }
      }

      // Set the ground class rgb
      if (max_log_odd_ground > 0) {
        (*ground_class_)(i) = packRGB(max_class_rgb_ground[0], max_class_rgb_ground[1], max_class_rgb_ground[2]);
      }

      // Set the obstacle class rgb
      if (max_log_odd_obstacle > 0) {
        (*obstacle_class_)(i) = packRGB(max_class_rgb_obstacle[0], max_class_rgb_obstacle[1], max_class_rgb_obstacle[2]);
      }

    }

    last_update_stamp_ = msg->header.stamp;
  }

  void applyFilterChain() {

    RCLCPP_INFO(this->get_logger(), "%d PC updates", pc_updates_);
    pc_updates_ = 0;

    // Map Iteration (iterate over whole map)
    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      const size_t i = it.getLinearIndex();

      // If min height is NaN, use value from the old layer
      if (std::isnan((*min_height_)(i))) {
        (*min_height_)(i) = (*min_height_old_)(i);
      }

      // Set the obstacle layer depending on the zone value
      if((*obstacle_zone_)(i) > 0)
      {
        (*obstacle_)(i) = 1000;
      }
      else
      {
        (*obstacle_)(i) = 0;
      }

      // Set the sky layer depending on the zone value
      if((*sky_zone_)(i) > 0)
      {
        (*sky_)(i) = 1000;
      }
      else
      {
        (*sky_)(i) = 0;
      }
    }

    // Measure Map Iteration Time
    auto map_iteration_time = std::chrono::steady_clock::now();


    // Fill the obstacle zone
    //markAlphaShapeObstacleClusters(map_, "obstacle_zone", 2, this->get_logger());
    morphologicalClose3x3(map_, "obstacle");
    morphologicalClose3x3(map_, "sky_map");

    // Measure Obstacle Zone Time
    auto obstacle_zone_time = std::chrono::steady_clock::now();
    RCLCPP_DEBUG(this->get_logger(), "Obstacle zone took %f ms", std::chrono::duration<double, std::milli>(obstacle_zone_time - map_iteration_time).count());

    // Apply filter chain.
    grid_map::GridMap min_height_filtered;
    if (!filterChain_.update(map_, min_height_filtered)) {
      RCLCPP_ERROR(this->get_logger(), "Could not update the grid map filter chain!");
      return;
    }

    // Measure Filter Chain Time
    //auto filter_chain_time = std::chrono::steady_clock::now();
    //RCLCPP_DEBUG(this->get_logger(), "Filter chain took %f ms", std::chrono::duration<double, std::milli>(filter_chain_time - obstacle_zone_time).count());

    // Copy min_height_smooth from min_height_filtered to map_
    map_["min_height_smooth"] = min_height_filtered["min_height_smooth"];

    grid_map_msgs::msg::GridMap map_msg;
    map_msg = *grid_map::GridMapRosConverter::toMessage(map_);
    map_msg.header.stamp = this->last_update_stamp_;
    grid_map_pub_->publish(map_msg);
    map_["min_height_old"] = map_["min_height"];
    map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
    // map_["obstacle_zone"].setConstant(std::numeric_limits<float>::quiet_NaN());
  }

};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticGridMapper>());
  rclcpp::shutdown();
  return 0;
}