cmake_minimum_required(VERSION 3.5)
project(stereo_image_proc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)

# See note in image_proc/CMakeLists.txt
# add_definitions(-DOPENCV_TRAITS_ENABLE_DEPRECATED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/${PROJECT_NAME}/stereo_processor.cpp
  src/${PROJECT_NAME}/disparity_node.cpp
  src/${PROJECT_NAME}/point_cloud_node.cpp
  src/${PROJECT_NAME}/rect_preprocess_node.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

# Register individual components and also build standalone nodes for each
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "stereo_image_proc::DisparityNode"
  EXECUTABLE disparity_node
)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "stereo_image_proc::PointCloudNode"
  EXECUTABLE point_cloud_node
)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "stereo_image_proc::RectPreprocessNode"
  EXECUTABLE rect_preprocess_node
)

ament_auto_package(
    INSTALL_TO_SHARE launch
    USE_SCOPED_HEADER_INSTALL_DIR
    )