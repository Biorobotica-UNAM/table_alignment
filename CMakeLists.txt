cmake_minimum_required(VERSION 3.8)
project(table_alignment)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic) #Wpedantic sirve para que el código sea más estricto
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(table_segmenter_node src/table_segmenter_node.cpp)

install(TARGETS 
  table_segmenter_node
  DESTINATION lib/${PROJECT_NAME})

ament_target_dependencies(table_segmenter_node
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
  geometry_msgs
  tf2_ros
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
