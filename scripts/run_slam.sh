#!/bin/bash
source ~/ros2_ws/install/setup.bash
export DISPLAY=:1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

ros2 run orbslam3 rgbd \
  ~/ros2_ws/src/orbslam3_ros2/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  ~/ros2_ws/src/orbslam3_ros2/ORB_SLAM3/Examples/RGB-D/RealSense_D435i.yaml \
  --ros-args \
  -r camera/color/image_raw:=/local/color/image_raw \
  -r camera/depth/image_rect_raw:=/local/depth/image_rect_raw
