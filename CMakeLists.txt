cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(semantic_grid_mapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_demos REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(GMP REQUIRED)
find_package(filters REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${GMP_INCLUDE_DIRS}
)

add_executable(semantic_grid_mapper src/semantic_grid_mapper.cpp)

ament_target_dependencies(semantic_grid_mapper
  rclcpp
  sensor_msgs
  std_msgs
  grid_map_ros
  grid_map_core
  grid_map_msgs
  grid_map_demos
  pcl_conversions
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  pcl_ros
  filters
)

# Add this line to link GMP
target_link_libraries(semantic_grid_mapper
  ${PCL_LIBRARIES}
  ${GMP_LIBRARIES}
  CGAL::CGAL
)

install(TARGETS semantic_grid_mapper
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
