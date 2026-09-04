#ifndef DLIO_ODOM_HPP_ 
#define DLIO_ODOM_HPP_ 

#include "dlio/dlio.h"

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <string>

// BOOST
#include <boost/format.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/adjacent_filtered.hpp>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

namespace dlio
{

class OdomNode : public rclcpp::Node
{
public:
  explicit OdomNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~OdomNode();

  void start();

private:

  struct State;
  struct ImuMeas {
    double stamp;
    double dt;
    // Measurements transformed into base_link but not bias-corrected. A single
    // bias snapshot is applied to a copied integration interval.
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  };

  struct ImuCalibrationSample {
    double stamp;
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  };

  enum class ImuIntegrationStatus : uint8_t {
    kSuccess,
    kInvalidRange,
    kImuTimeout,
    kHistoryUnavailable,
    kImuGap,
    kIncomplete,
  };

  enum class RecoveryState : uint8_t {
    kTracking,
    kReseedPending,
    kConfirming,
  };

  struct ImuIntegrationResult {
    ImuIntegrationStatus status = ImuIntegrationStatus::kInvalidRange;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
    Eigen::Vector3f anchor_velocity = Eigen::Vector3f::Zero();
    double oldest_imu_stamp = 0.0;
    double newest_imu_stamp = 0.0;
    std::string reason;

    bool ok() const { return status == ImuIntegrationStatus::kSuccess; }
  };

  void getParams();

  void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc);
  void callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu);

  void publishPose();
  void publishScanOdometry();
  void publishDiagnostics();
  nav_msgs::msg::Odometry makeOdometryMessage(
    const State& state_snapshot, const rclcpp::Time& stamp, double covariance_scale) const;

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp);

  void getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc);
  bool preprocessPoints();
  bool deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  bool getNextPose();
  bool imuMeasFromTimeRange(boost::circular_buffer<ImuMeas>& imu_snapshot,
                            double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  ImuIntegrationResult
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it,
                         std::size_t anchor_index, Eigen::Vector3f& anchor_velocity);
  void propagateGICP();

  void propagateState();
  void updateState();

  void setAdaptiveParams();
  void setKeyframeCloud();

  void computeMetrics();
  void computeSpaciousness();
  void computeDensity();

  sensor_msgs::msg::Imu::SharedPtr transformImu(
    const sensor_msgs::msg::Imu::SharedPtr& imu, double dt);

  void rejectScan(const ImuIntegrationResult& result, bool reanchor);
  void rejectScan(
    const std::string& reason, bool preserve_pending_correction = false);
  void markScanAccepted();
  void markOdometryUnhealthy(const std::string& reason);
  void setPredictionAnchor(
    const Eigen::Matrix4f& pose, const Eigen::Vector3f& velocity, double stamp);
  void resetPredictionAnchorToTrustedPose(double stamp);
  void scheduleRecovery(uint64_t rejected_count);
  bool submapReadyForRecovery();
  void reseedLocalMap();
  bool recoveryRegistrationConfirmed();
  const char* recoveryStateName() const;
  bool odometryOutputHealthy() const;
  void setOdometryCovariance(
    nav_msgs::msg::Odometry& message, double stamp_seconds, double scale) const;
  bool updateImuCalibration(
    double stamp, const Eigen::Vector3f& ang_vel, const Eigen::Vector3f& lin_accel);
  void updateOnlineGyroBias(
    double stamp, double dt, const Eigen::Vector3f& ang_vel,
    const Eigen::Vector3f& lin_accel);
  void updateAcceptedLidarMotion();
  bool validateRegistrationQuality(std::string& reason);
  bool correctionNeedsConfirmation(
    const Eigen::Matrix4f& correction, double correction_distance,
    double correction_rotation_deg, std::string& reason);
  void recordAcceptedRegistrationQuality();

  void updateKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames);
  void buildSubmap(State vehicle_state);
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void debug();

  rclcpp::TimerBase::SharedPtr publish_timer;
  rclcpp::TimerBase::SharedPtr diagnostics_timer;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::CallbackGroup::SharedPtr lidar_cb_group, imu_cb_group;

  // Publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr scan_odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr kf_pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_cloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub;

  // TF
  std::shared_ptr<tf2_ros::TransformBroadcaster> br;
  bool publish_odom_tf_;
  bool publish_sensor_tf_;

  // ROS Msgs
  nav_msgs::msg::Odometry odom_ros;
  geometry_msgs::msg::PoseStamped pose_ros;
  nav_msgs::msg::Path path_ros;
  geometry_msgs::msg::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
  double length_traversed;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<rclcpp::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;

  // Sensor Type
  dlio::SensorType sensor;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  // Keyframes
  pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  rclcpp::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  double scan_dt;
  std::vector<double> comp_times;
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;
  Eigen::Quaternionf q_final;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;

  // IMU
  double prev_imu_stamp;
  double last_received_imu_stamp_;
  double imu_dp, imu_dq_deg;
  ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;
  std::atomic<int64_t> latest_imu_stamp_ns_{0};
  std::atomic<double> last_accepted_scan_stamp_{0.0};
  std::atomic<int64_t> last_accepted_steady_ns_{0};
  std::atomic<bool> temporal_reanchor_pending_{false};
  std::atomic<bool> odom_ready_{false};
  std::atomic<bool> imu_healthy_{false};
  std::atomic<bool> scan_healthy_{false};
  std::atomic<uint64_t> consecutive_rejected_scans_{0};
  std::atomic<int> recovery_scans_remaining_{0};
  std::atomic<RecoveryState> recovery_state_{RecoveryState::kTracking};
  std::atomic<uint64_t> recovery_reseed_count_{0};
  std::atomic<int> recovery_confirmation_count_{0};
  Eigen::Vector3f prediction_anchor_position_ = Eigen::Vector3f::Zero();
  Eigen::Quaternionf prediction_anchor_orientation_ = Eigen::Quaternionf::Identity();
  Eigen::Vector3f prediction_anchor_velocity_ = Eigen::Vector3f::Zero();
  double prediction_anchor_stamp_ = 0.0;
  std::atomic<double> prediction_anchor_diagnostic_stamp_{0.0};
  mutable std::mutex health_mutex_;
  std::string health_reason_;

  std::deque<ImuCalibrationSample> imu_calibration_samples_;
  std::deque<ImuCalibrationSample> online_bias_samples_;
  Eigen::Vector3f startup_gyro_bias_ = Eigen::Vector3f::Zero();

  bool accepted_lidar_pose_available_ = false;
  Eigen::Vector3f previous_accepted_lidar_position_ = Eigen::Vector3f::Zero();
  Eigen::Quaternionf previous_accepted_lidar_orientation_ = Eigen::Quaternionf::Identity();
  double previous_accepted_lidar_stamp_ = 0.0;
  std::atomic<double> accepted_lidar_linear_speed_{
    std::numeric_limits<double>::infinity()};
  std::atomic<double> accepted_lidar_angular_speed_{
    std::numeric_limits<double>::infinity()};

  std::deque<double> registration_error_history_;
  std::deque<double> registration_condition_history_;
  std::atomic<double> last_correspondence_ratio_{0.0};
  std::atomic<double> last_normalized_registration_error_{0.0};
  std::atomic<double> last_hessian_condition_{0.0};
  bool pending_correction_valid_ = false;
  Eigen::Matrix4f pending_correction_ = Eigen::Matrix4f::Identity();
  int pending_correction_count_ = 0;

  static bool comparatorImu(ImuMeas m1, ImuMeas m2) {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  }; State state;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;
  Pose imuPose;

  // Metrics
  struct Metrics {
    std::vector<float> spaciousness;
    std::vector<float> density;
  }; Metrics metrics;

  std::string cpu_type;
  std::vector<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;
  int num_threads_;

  bool debug_enabled_;

  std::string pointcloud_qos_reliability_;
  int pointcloud_qos_depth_;
  std::string imu_qos_reliability_;
  int imu_qos_depth_;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  double submap_concave_alpha_;

  bool densemap_filtered_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_calibration_min_samples_;
  double imu_calibration_gyro_stddev_max_;
  double imu_calibration_gyro_mean_max_;
  double imu_calibration_accel_norm_stddev_max_;
  double imu_calibration_accel_gravity_tolerance_;
  bool online_gyro_bias_enabled_;
  double online_gyro_bias_stationary_time_;
  double online_gyro_bias_lidar_linear_max_;
  double online_gyro_bias_lidar_angular_max_deg_;
  double online_gyro_bias_time_constant_;
  double online_gyro_bias_max_delta_;
  int imu_buffer_size_;
  double imu_max_gap_seconds_;
  double imu_wait_timeout_seconds_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;
  double gicp_max_correction_distance_;
  double gicp_max_correction_rotation_deg_;
  double gicp_min_correspondence_ratio_;
  int gicp_quality_history_size_;
  int gicp_quality_warmup_scans_;
  double gicp_max_normalized_error_factor_;
  double gicp_max_hessian_condition_;
  double gicp_max_hessian_condition_factor_;
  double gicp_soft_correction_distance_;
  double gicp_soft_correction_rotation_deg_;
  int gicp_correction_confirmation_scans_;
  double gicp_correction_consistency_distance_;
  double gicp_correction_consistency_rotation_deg_;

  std::vector<double> pose_covariance_diagonal_;
  std::vector<double> twist_covariance_diagonal_;
  double covariance_degraded_after_seconds_;
  double covariance_pose_position_rate_;
  double covariance_pose_yaw_rate_;
  double covariance_twist_linear_rate_;
  double covariance_twist_yaw_rate_;
  double covariance_recovery_scale_;
  int covariance_recovery_scans_;
  double odom_publish_max_scan_age_;
  bool recovery_enabled_;
  int recovery_rejected_scans_;
  int recovery_confirmation_scans_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

};

} // namespace dlio

#endif // DLIO_ODOM_HPP_ 
