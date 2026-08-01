###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED )
find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES       
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)
include_directories(SYSTEM)

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
set_target_properties(nanoflann PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nanoflann ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
set_target_properties(nano_gicp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)

# ===================================================================
# Composable odometry component (dlio::OdomNode)
# ===================================================================
add_library(dlio_odom_component SHARED
  src/dlio/odom.cpp
)

target_link_libraries(dlio_odom_component
  ${PCL_LIBRARIES}
  ${OpenMP_LIBS}
  Threads::Threads
  nano_gicp
  rclcpp::rclcpp
  rclcpp_components::component
  tf2_ros::tf2_ros
  pcl_conversions::pcl_conversions
  nav_msgs::nav_msgs__rosidl_typesupport_cpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
)

# Register the component (matches RCLCPP_COMPONENTS_REGISTER_NODE in odom.cpp)
rclcpp_components_register_nodes(dlio_odom_component "dlio::OdomNode")

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cpp src/dlio/odom.cpp)
target_link_libraries(dlio_odom_node
  ${PCL_LIBRARIES}
  ${OpenMP_LIBS}
  Threads::Threads
  nano_gicp
  rclcpp::rclcpp
  rclcpp_components::component
  tf2_ros::tf2_ros
  pcl_conversions::pcl_conversions
  nav_msgs::nav_msgs__rosidl_typesupport_cpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
)

# Mapping Node
add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
target_link_libraries(dlio_map_node
  ${PCL_LIBRARIES}
  ${cpp_typesupport_target}
  rclcpp::rclcpp
  pcl_conversions::pcl_conversions
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} nano_gicp nanoflann)
ament_export_dependencies(rclcpp rclcpp_components std_msgs sensor_msgs geometry_msgs pcl_ros)
ament_package()

# Binaries
install( TARGETS dlio_odom_node dlio_map_node
         DESTINATION lib/${PROJECT_NAME} 
)

install(
  TARGETS
    dlio_odom_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install( DIRECTORY cfg launch
         DESTINATION share/${PROJECT_NAME} 
)

add_compile_options(-std=c++14)
link_directories(${PCL_LIBRARY_DIRS})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

target_compile_options(dlio_odom_node PRIVATE ${OpenMP_FLAGS})
target_compile_options(dlio_map_node PRIVATE ${OpenMP_FLAGS})
target_compile_options(dlio_odom_component PRIVATE ${OpenMP_FLAGS})
