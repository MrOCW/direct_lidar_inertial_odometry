# Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction

#### [[ IEEE ICRA ](https://ieeexplore.ieee.org/document/10160508)] [[ arXiv ](https://arxiv.org/abs/2203.03749)] [[ Video ](https://www.youtube.com/watch?v=4-oXjG8ow10)] [[ Presentation ](https://www.youtube.com/watch?v=Hmiw66KZ1tU)]

DLIO is a new lightweight LiDAR-inertial odometry algorithm with a novel coarse-to-fine approach in constructing continuous-time trajectories for precise motion correction. It features several algorithmic improvements over its predecessor, [DLO](https://github.com/vectr-ucla/direct_lidar_odometry), and was presented at the IEEE International Conference on Robotics and Automation (ICRA) in London, UK in 2023.

<br>
<p align='center'>
    <img src="./doc/img/dlio.png" alt="drawing" width="720"/>
</p>

## Main changes from the original repo
- Bugfixes
- Support Ubuntu 24.04, ROS2 Jazzy (note in Setup section)
- Remove detached threads causing crashes and publish messages synchronously
- Odometry node now supports ComposableNode

## Setup
This fork was only tested with the following:
- Jetson Xavier NX
    - Jetpack 5
    - ROS2 Jazzy docker image
- Mid-360 Lidar

```sh
sudo apt install libomp-dev libpcl-dev libeigen3-dev ros-jazzy-pcl-ros
```

After compilation, execute via
```sh
ros2 launch direct_lidar_inertial_odometry dlio_composable.launch.py \
  rviz:={true, false} \
  pointcloud_topic:=/lidar \
  imu_topic:=/imu
  
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  rviz:={true, false} \
  pointcloud_topic:=/lidar \
  imu_topic:=/imu
```

