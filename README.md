# Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction

#### [[ IEEE ICRA ](https://ieeexplore.ieee.org/document/10160508)] [[ arXiv ](https://arxiv.org/abs/2203.03749)] [[ Video ](https://www.youtube.com/watch?v=4-oXjG8ow10)] [[ Presentation ](https://www.youtube.com/watch?v=Hmiw66KZ1tU)]

DLIO is a new lightweight LiDAR-inertial odometry algorithm with a novel coarse-to-fine approach in constructing continuous-time trajectories for precise motion correction. It features several algorithmic improvements over its predecessor, [DLO](https://github.com/vectr-ucla/direct_lidar_odometry), and was presented at the IEEE International Conference on Robotics and Automation (ICRA) in London, UK in 2023.

<br>
<p align='center'>
    <img src="./doc/img/dlio.png" alt="drawing" width="720"/>
</p>

## Changes in this fork

- Fixes for crashes caused by detached worker threads
- Synchronous ROS message publication
- A composable `dlio::OdomNode`
- Livox MID-360 configuration
- ROS 2 Jazzy support on the `jazzy` branch
- ROS 2 Lyrical support on the `lyrical` branch

## Supported setup

This fork has been tested with:

- NVIDIA Jetson Xavier NX
- Livox MID-360 LiDAR and IMU
- ROS2 Lyrical Docker image

## Build for ROS 2 Lyrical

Select the Lyrical branch and install dependencies with rosdep:

```bash
git switch lyrical
source /opt/ros/lyrical/setup.bash
rosdep install --from-paths . --ignore-src -r -y --rosdistro lyrical
```

From the ROS workspace root, build the package:

```bash
colcon build --packages-select direct_lidar_inertial_odometry \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Launch

Run the composable odometry node with the MID-360 topics:

```bash
ros2 launch direct_lidar_inertial_odometry dlio_composable.launch.py \
  rviz:=true \
  pointcloud_topic:=/livox/lidar \
  imu_topic:=/livox/imu
```

Or run the standalone odometry and mapping nodes:

```bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py \
  rviz:=true \
  pointcloud_topic:=/livox/lidar \
  imu_topic:=/livox/imu
```

Set `rviz:=false` for headless operation. Odometry is published on
`/dlio/odom_node/odom`, with the trajectory on `/dlio/odom_node/path` and the
deskewed cloud on `/dlio/odom_node/pointcloud/deskewed`.

## TF ownership

`publish/odom_tf` controls whether the odometry node broadcasts the dynamic
`odom -> base_link` transform.