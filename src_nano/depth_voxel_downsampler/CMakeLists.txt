cmake_minimum_required(VERSION 3.8)
project(depth_voxel_downsampler)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Jetson 등 ARM 환경 호환 위해 -march=native는 제거 권장
add_compile_options(-O3 -Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS filters)
find_package(pcl_conversions REQUIRED)
find_package(rcl_interfaces REQUIRED)  # ★ 추가

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_executable(voxel_downsampler_node src/voxel_downsampler_node.cpp)
ament_target_dependencies(voxel_downsampler_node
  rclcpp sensor_msgs pcl_conversions rcl_interfaces)  # ★ rcl_interfaces 연결
target_link_libraries(voxel_downsampler_node ${PCL_LIBRARIES})

install(TARGETS voxel_downsampler_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
