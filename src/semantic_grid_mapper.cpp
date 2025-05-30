#include "semantic_grid_mapper/semantic_grid_mapper.hpp"

#include "cluster.cpp"



SemanticGridMapper::SemanticGridMapper()
      : Node("semantic_grid_mapper"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_), filterChain_("grid_map::GridMap") 
{

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

  // Probabilistic layer for each class (log odd format) and each area (ground, obstacle, sky)
  for (auto name : class_names_) {
    map_.add(name + "_ground");
    map_.add(name + "_obstacle");
    // map_.add(name + "_sky");
  }

  // Visualization layer for each class
  // for (auto name : class_names_) {
  //   map_.add(name + "_rgb");
  // }

  // Hit count layer for each class
  for (auto name : class_names_) {
    map_.add(name + "_ground_hit");
    map_.add(name + "_obstacle_hit");
    // map_.add(name + "_sky_hit");
  }

  // Probability layers (% format 0..1)
  for (auto name : class_names_) {
    map_.add(name + "_ground_prob");
    map_.add(name + "_obstacle_prob");
    // map_.add(name + "_sky_prob");
  }

  // Independent Layer for each class. The above probabilities have cross-class dependencies.
  // For example if a cell has a high probability to be tree-foliage, the probability for tree-trunk is low
  // even though there might be hits in the cell. This is a problem if we wan't to extract the info whether there is a tree trunk because
  // the info could get lost due to many hits for tree-foliage.
  // for (auto name : class_names_) {
  //   map_.add(name + "_idp");
  //   map_[name + "_idp"].setConstant(0.0);
  // }

  // Visualization layer which combines all classes
  map_.add("ground_class");
  map_["ground_class"].setConstant(-1.0);
  map_.add("obstacle_class");
  map_["obstacle_class"].setConstant(-1.0);
  // map_.add("sky_class");
  // map_["sky_class"].setConstant(-1.0);

  // Initialize the map
  map_.setGeometry(grid_map::Length(length_, height_), resolution_,
                    grid_map::Position(0.0, 0.0));
  map_.setFrameId(map_frame_id_);

  for (const auto &name : class_names_) {
    map_[name + "_ground"].setConstant(0.0); // log-odds 0 => p = 0.5
    map_[name + "_obstacle"].setConstant(0.0); // log-odds 0 => p = 0.5
    // map_[name + "_sky"].setConstant(0.0); // log-odds 0 => p = 0.5
  }

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

  cls_.reserve(class_names_.size());
  for (auto const & name : class_names_) {
    cls_.push_back({
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
      });
  }

  update_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&SemanticGridMapper::updateMap, this)
    // /* use_steady_clock = */ false  // default
  );

  RCLCPP_INFO(this->get_logger(), "Semantic Grid Mapper initialized.");
}

double SemanticGridMapper::update_log_odds(double old_l, double meas_l) {
  // If new cell (not yet initialized, only new value)
  if (std::isnan(old_l)) {
    return std::clamp(meas_l, log_odd_min_, log_odd_max_);
  }

  // Update log-odds and clamp to range [-10, 10]
  double updated_l = old_l + meas_l;
  return std::clamp(updated_l, log_odd_min_, log_odd_max_);
}

void SemanticGridMapper::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

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
      float float_rgb = map_.at("ground_class", idx);
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

void SemanticGridMapper::semanticPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

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
  
  // Reset Hit Counts
  for (auto name : class_names_) {
    map_[name + "_ground_hit"].setConstant(0.0);
    map_[name + "_obstacle_hit"].setConstant(0.0);
    // map_[name + "_sky_hit"].setConstant(0.0);
  }

  // Measure Layer Initialization Time
  auto layer_initialization_time = std::chrono::steady_clock::now();
  RCLCPP_DEBUG(this->get_logger(), "Layer initialization took %f ms", std::chrono::duration<double, std::milli>(layer_initialization_time - pointcloud_transform_time).count());

  // Point Iteration
  for (const auto &point : transformed_cloud.points) 
  {
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
    // TODO: Remove this
    grid_map::Index idx;
    map_.getIndex(pos, idx);
    float min_height = map_.at("min_height", idx);
    if (std::isnan(min_height) || point.z < min_height) {
      map_.at("min_height", idx) = point.z;
    }
    
    // Update class hit count
    std::string cls = color_to_class_[color];
    if(map_.exists("min_height_smooth")) {

      // Ground Layer
      if(point.z < map_.at("min_height_smooth", idx) + max_veg_height_) {
        map_.at(cls + "_ground_hit", idx) += 1.0;
      }

      // Obstacle Layer
      else if(point.z >= map_.at("min_height_smooth", idx) + max_veg_height_ 
      && point.z < map_.at("min_height_smooth", idx) + robot_height_) {
        map_.at(cls + "_obstacle_hit", idx) += 1.0;
      }

      // // Sky Layer
      // else if(point.z >= map_.at("min_height_smooth", idx) + robot_height_) {
      //   map_.at(cls + "_sky_hit", idx) += 1.0;
      // }
    }

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
}

void SemanticGridMapper::updateMap() 
{

  // Check if the map layer are already available
  if (!map_.exists("min_height") || !map_.exists("min_height_old") 
      || !map_.exists("obstacle_zone") || !map_.exists("vegetation_zone")) {
    RCLCPP_WARN(this->get_logger(), "Map layers not initialized yet, skipping update.");
    return;
  }

  // Measure Point Iteration Time
  auto start_time = std::chrono::steady_clock::now();

  // Reset probabilities
  for (auto name : class_names_) {
    map_[name + "_ground_prob"].setConstant(0.0);
    map_[name + "_obstacle_prob"].setConstant(0.0);
  }

  // Map Iteration 1
  for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) 
  {
    grid_map::Index index = *it;

    // Calculate the total hits across all classes at this cell
    float total_ground_hits = 0.0;
    float total_obstacle_hits = 0.0;
    // float total_sky_hits = 0.0;

    for(auto &cls : cls_) {
      // Ground Layer
      total_ground_hits += (*(cls.hit_ground))(index(0), index(1));
      // Obstacle Layer
      total_obstacle_hits += (*(cls.hit_obstacle))(index(0), index(1));
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
    for (auto &cls : cls_) {
      
      // Ground Layer
      float prob_ground = total_ground_hits ? (*(cls.hit_ground))(index(0), index(1)) / total_ground_hits : 0.0;
      (*(cls.prob_ground))(index(0), index(1)) = prob_ground;

      // Obstacle Layer
      float prob_obstacle = total_obstacle_hits ? (*(cls.hit_obstacle))(index(0), index(1)) / total_obstacle_hits : 0.0;
      (*(cls.prob_obstacle))(index(0), index(1)) = prob_obstacle;

      // // Sky Layer
      // (*(cls.prob_sky))(index(0), index(1)) = total_sky_hits ? cls.hit_sky->at(index) / total_sky_hits : 0.0;

      double log_odds_ground = update_log_odds(
          (*(cls.hist_ground))(index(0), index(1)), prob_to_log_odds(prob_ground));

      double log_odds_obstacle = update_log_odds(
          (*(cls.hist_obstacle))(index(0), index(1)), prob_to_log_odds(prob_obstacle));

      (*(cls.hist_ground))(index(0), index(1)) = log_odds_ground;
      (*(cls.hist_obstacle))(index(0), index(1)) = log_odds_obstacle;

      if(log_odds_ground > max_log_odd_ground) { 
        max_log_odd_ground = log_odds_ground; 
        max_class_rgb_ground = cls.rgb; 
      }
      if(log_odds_obstacle > max_log_odd_obstacle) { 
        max_log_odd_obstacle = log_odds_obstacle; 
        max_class_rgb_obstacle = cls.rgb; 
      }

    }

    // Set obstacle zone to nan if too few (noise)
    if (map_.at("obstacle_zone", index) < 5.0) {
      map_.at("obstacle_zone", index) = std::numeric_limits<float>::quiet_NaN();
    }

    // Set the ground class rgb
    if (max_log_odd_ground > 0) {
      map_.at("ground_class", index) = packRGB(max_class_rgb_ground[0], max_class_rgb_ground[1], max_class_rgb_ground[2]);
    }

    // Set the obstacle class rgb
    if (max_log_odd_obstacle > 0) {
      map_.at("obstacle_class", index) = packRGB(max_class_rgb_obstacle[0], max_class_rgb_obstacle[1], max_class_rgb_obstacle[2]);
    }

    // If min height is NaN, use value from the old layer
    if (std::isnan(map_.at("min_height", index))) {
      map_.at("min_height", index) = map_.at("min_height_old", index);
    }
  }

  // Measure Map Iteration Time
  auto map_iteration_time = std::chrono::steady_clock::now();
  RCLCPP_DEBUG(this->get_logger(), "Map iteration took %f ms", std::chrono::duration<double, std::milli>(map_iteration_time - start_time).count());

  // Fill the obstacle zone
  std::string obstacle_zone = "obstacle_zone";


  markAlphaShapeObstacleClusters(map_, obstacle_zone, 2);

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
  auto filter_chain_time = std::chrono::steady_clock::now();
  RCLCPP_DEBUG(this->get_logger(), "Filter chain took %f ms", std::chrono::duration<double, std::milli>(filter_chain_time - obstacle_zone_time).count());

  // Copy min_height_smooth from min_height_filtered to map_
  if(!map_.exists("min_height_smooth")) {
    map_.add("min_height_smooth");
  }
  map_["min_height_smooth"] = min_height_filtered["min_height_smooth"];

  // Remove unnecessary layers from min_height_filtered
  for(const auto &name : class_names_) {
    if (min_height_filtered.exists(name + "_ground")) {
      min_height_filtered.erase(name + "_ground");
    }
    if (min_height_filtered.exists(name + "_obstacle")) {
      min_height_filtered.erase(name + "_obstacle");
    }
    if (min_height_filtered.exists(name + "_sky")) {
      min_height_filtered.erase(name + "_sky");
    }
    if (min_height_filtered.exists(name + "_ground_hit")) {
      min_height_filtered.erase(name + "_ground_hit");
    }
    if (min_height_filtered.exists(name + "_obstacle_hit")) {
      min_height_filtered.erase(name + "_obstacle_hit");
    }
    if (min_height_filtered.exists(name + "_sky_hit")) {
      min_height_filtered.erase(name + "_sky_hit");
    }
    if (min_height_filtered.exists(name + "_ground_prob")) {
      min_height_filtered.erase(name + "_ground_prob");
    }
    if (min_height_filtered.exists(name + "_obstacle_prob")) {
      min_height_filtered.erase(name + "_obstacle_prob");
    }
    if (min_height_filtered.exists(name + "_sky_prob")) {
      min_height_filtered.erase(name + "_sky_prob");
    }
  }

  grid_map_msgs::msg::GridMap map_msg;
  map_msg = *grid_map::GridMapRosConverter::toMessage(min_height_filtered);
  map_msg.header.stamp = last_cloud_stamp_;
  grid_map_pub_->publish(map_msg);
  map_["min_height_old"] = map_["min_height"];
  map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
  map_["obstacle_zone"].setConstant(std::numeric_limits<float>::quiet_NaN());

  // Measure Publishing Time
  auto publishing_time = std::chrono::steady_clock::now();
  RCLCPP_INFO(this->get_logger(), "Publishing took %f ms", std::chrono::duration<double, std::milli>(publishing_time - start_time).count());
}


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticGridMapper>());
  rclcpp::shutdown();
  return 0;
}