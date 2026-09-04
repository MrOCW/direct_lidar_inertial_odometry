/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/odom.hpp"
#include "dlio/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/SVD>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include "rclcpp/qos.hpp"
namespace dlio
{

namespace
{
class ScopeExit
{
public:
  explicit ScopeExit(std::function<void()> callback)
  : callback_(std::move(callback)) {}

  ~ScopeExit() { run(); }

  void run()
  {
    if (active_) {
      active_ = false;
      callback_();
    }
  }

private:
  std::function<void()> callback_;
  bool active_ = true;
};

rclcpp::QoS makeSensorQos(int depth, const std::string & reliability)
{
  if (depth <= 0) {
    throw std::invalid_argument("Sensor QoS depth must be greater than zero");
  }

  rclcpp::QoS qos{rclcpp::KeepLast(static_cast<size_t>(depth))};
  qos.durability_volatile();
  if (reliability == "best_effort") {
    qos.best_effort();
  } else if (reliability == "reliable") {
    qos.reliable();
  } else {
    throw std::invalid_argument(
      "Sensor QoS reliability must be 'best_effort' or 'reliable', got '" + reliability + "'");
  }
  return qos;
}
}  // namespace
  
OdomNode::OdomNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("dlio_odom_node", options) {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  this->lidar_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto lidar_sub_opt = rclcpp::SubscriptionOptions();
  lidar_sub_opt.callback_group = this->lidar_cb_group;
  const auto pointcloud_qos = makeSensorQos(
    this->pointcloud_qos_depth_, this->pointcloud_qos_reliability_);
  this->lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("pointcloud", pointcloud_qos,
      std::bind(&OdomNode::callbackPointCloud, this, std::placeholders::_1), lidar_sub_opt);

  this->imu_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  auto imu_sub_opt = rclcpp::SubscriptionOptions();
  imu_sub_opt.callback_group = this->imu_cb_group;
  const auto imu_qos = makeSensorQos(this->imu_qos_depth_, this->imu_qos_reliability_);
  this->imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu", imu_qos,
      std::bind(&OdomNode::callbackImu, this, std::placeholders::_1), imu_sub_opt);

  this->odom_pub     = this->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  this->scan_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("odom_scan", 1);
  this->pose_pub     = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose", 1);
  this->path_pub     = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  this->kf_pose_pub  = this->create_publisher<geometry_msgs::msg::PoseArray>("kf_pose", 1);
  this->kf_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("kf_cloud", 1);
  this->deskewed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("deskewed", 1);
  this->diagnostics_pub =
    this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);

  this->br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  this->publish_timer = this->create_wall_timer(std::chrono::duration<double>(0.01), 
      std::bind(&OdomNode::publishPose, this));
  this->diagnostics_timer = this->create_wall_timer(
      std::chrono::seconds(1), std::bind(&OdomNode::publishDiagnostics, this));

  this->health_reason_ = "waiting for the first accepted LiDAR scan";

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->prev_imu_stamp = 0.;
  this->last_received_imu_stamp_ = 0.;
  this->startup_gyro_bias_ = this->state.b.gyro;

  this->original_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->deskewed_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->current_scan = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed = 0.;

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

}

OdomNode::~OdomNode() {}

void OdomNode::getParams() {

  // Version
  declare_param(this, "version", this->version_, "0.0.0");

  // Frames
  declare_param(this, "frames/odom", this->odom_frame, "odom");
  declare_param(this, "frames/baselink", this->baselink_frame, "base_link");
  declare_param(this, "frames/lidar", this->lidar_frame, "lidar");
  declare_param(this, "frames/imu", this->imu_frame, "imu");

  // TF publication
  declare_param(this, "publish/odom_tf", this->publish_odom_tf_, true);
  declare_param(this, "publish/sensor_tf", this->publish_sensor_tf_, true);

  // Terminal debug dashboard
  declare_param(this, "debug/enable", this->debug_enabled_, false);

  // Keep middleware queues bounded so stale sensor data is never replayed in a burst.
  declare_param(this, "pointcloud/qos/reliability", this->pointcloud_qos_reliability_,
    std::string("best_effort"));
  declare_param(this, "pointcloud/qos/depth", this->pointcloud_qos_depth_, 2);
  declare_param(this, "imu/qos/reliability", this->imu_qos_reliability_,
    std::string("best_effort"));
  declare_param(this, "imu/qos/depth", this->imu_qos_depth_, 50);

  // Deskew Flag
  declare_param(this, "pointcloud/deskew", this->deskew_, true);

  // Gravity
  declare_param(this, "odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  declare_param(this, "odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  declare_param(this, "odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  declare_param(this, "odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  declare_param(this, "odom/submap/keyframe/knn", this->submap_knn_, 10);
  declare_param(this, "odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  declare_param(this, "odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // Dense map resolution
  declare_param(this, "map/dense/filtered", this->densemap_filtered_, true);

  // Wait until movement to publish map
  declare_param(this, "map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  declare_param(this, "odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  declare_param(this, "pointcloud/voxelize", this->vf_use_, true);
  declare_param(this, "odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  declare_param(this, "adaptive", this->adaptive_params_, true);

  // Extrinsics
  std::vector<double> t_default{0., 0., 0.};
  std::vector<double> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<double> baselink2imu_t, baselink2imu_R;
  declare_param(this, "extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  declare_param(this, "extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2imu_R.begin(), baselink2imu_R.end()).data(), 3, 3);
  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<double> baselink2lidar_t, baselink2lidar_R;
  declare_param(this, "extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  declare_param(this, "extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(baselink2lidar_R.begin(), baselink2lidar_R.end()).data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  declare_param(this, "odom/imu/calibration/accel", this->calibrate_accel_, true);
  declare_param(this, "odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  declare_param(this, "odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  declare_param(this, "odom/imu/calibration/minSamples",
    this->imu_calibration_min_samples_, 400);
  declare_param(this, "odom/imu/calibration/gyroStddevMax",
    this->imu_calibration_gyro_stddev_max_, 0.005);
  declare_param(this, "odom/imu/calibration/gyroMeanMax",
    this->imu_calibration_gyro_mean_max_, 0.05);
  declare_param(this, "odom/imu/calibration/accelNormStddevMax",
    this->imu_calibration_accel_norm_stddev_max_, 0.08);
  declare_param(this, "odom/imu/calibration/accelGravityTolerance",
    this->imu_calibration_accel_gravity_tolerance_, 0.5);
  declare_param(this, "odom/imu/onlineGyroBias/enabled",
    this->online_gyro_bias_enabled_, true);
  declare_param(this, "odom/imu/onlineGyroBias/stationaryTime",
    this->online_gyro_bias_stationary_time_, 2.0);
  declare_param(this, "odom/imu/onlineGyroBias/lidarLinearMax",
    this->online_gyro_bias_lidar_linear_max_, 0.02);
  declare_param(this, "odom/imu/onlineGyroBias/lidarAngularMax",
    this->online_gyro_bias_lidar_angular_max_deg_, 0.5);
  declare_param(this, "odom/imu/onlineGyroBias/timeConstant",
    this->online_gyro_bias_time_constant_, 30.0);
  declare_param(this, "odom/imu/onlineGyroBias/maxDelta",
    this->online_gyro_bias_max_delta_, 0.05);
  declare_param(this, "odom/imu/bufferSize", this->imu_buffer_size_, 2000);
  declare_param(this, "odom/imu/maxGap", this->imu_max_gap_seconds_, 0.05);
  declare_param(this, "odom/imu/waitTimeout", this->imu_wait_timeout_seconds_, 0.05);

  std::vector<double> accel_default{0., 0., 0.}; std::vector<double> prior_accel_bias;
  std::vector<double> gyro_default{0., 0., 0.}; std::vector<double> prior_gyro_bias;

  declare_param(this, "odom/imu/approximateGravity", this->gravity_align_, true);
  declare_param(this, "imu/calibration", this->imu_calibrate_, true);
  declare_param(this, "imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  declare_param(this, "imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<double> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<double> imu_sm;

  declare_param(this, "imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(std::vector<float>(imu_sm.begin(), imu_sm.end()).data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  declare_param(this, "odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  declare_param(this, "odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  declare_param(this, "odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  declare_param(this, "odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  declare_param(this, "odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  declare_param(this, "odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  declare_param(this, "odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);
  declare_param(this, "odom/gicp/maxCorrectionDistance",
    this->gicp_max_correction_distance_, 1.0);
  declare_param(this, "odom/gicp/maxCorrectionRotation",
    this->gicp_max_correction_rotation_deg_, 45.0);
  declare_param(this, "odom/gicp/minCorrespondenceRatio",
    this->gicp_min_correspondence_ratio_, 0.20);
  declare_param(this, "odom/gicp/qualityHistorySize",
    this->gicp_quality_history_size_, 50);
  declare_param(this, "odom/gicp/qualityWarmupScans",
    this->gicp_quality_warmup_scans_, 20);
  declare_param(this, "odom/gicp/maxNormalizedErrorFactor",
    this->gicp_max_normalized_error_factor_, 3.0);
  declare_param(this, "odom/gicp/maxHessianCondition",
    this->gicp_max_hessian_condition_, 1.0e8);
  declare_param(this, "odom/gicp/maxHessianConditionFactor",
    this->gicp_max_hessian_condition_factor_, 10.0);
  declare_param(this, "odom/gicp/softCorrectionDistance",
    this->gicp_soft_correction_distance_, 0.06);
  declare_param(this, "odom/gicp/softCorrectionRotation",
    this->gicp_soft_correction_rotation_deg_, 1.0);
  declare_param(this, "odom/gicp/correctionConfirmationScans",
    this->gicp_correction_confirmation_scans_, 2);
  declare_param(this, "odom/gicp/correctionConsistencyDistance",
    this->gicp_correction_consistency_distance_, 0.02);
  declare_param(this, "odom/gicp/correctionConsistencyRotation",
    this->gicp_correction_consistency_rotation_deg_, 0.5);

  declare_param(this, "odom/covariance/pose/diagonal", this->pose_covariance_diagonal_,
    std::vector<double>{0.01, 0.01, 1000.0, 1000.0, 1000.0, 0.007615435});
  declare_param(this, "odom/covariance/twist/diagonal", this->twist_covariance_diagonal_,
    std::vector<double>{0.04, 0.04, 1000.0, 1000.0, 1000.0, 0.030461742});
  declare_param(this, "odom/covariance/degradedAfter",
    this->covariance_degraded_after_seconds_, 0.2);
  declare_param(this, "odom/covariance/posePositionRate",
    this->covariance_pose_position_rate_, 0.5);
  declare_param(this, "odom/covariance/poseYawRate",
    this->covariance_pose_yaw_rate_, 0.25);
  declare_param(this, "odom/covariance/twistLinearRate",
    this->covariance_twist_linear_rate_, 0.5);
  declare_param(this, "odom/covariance/twistYawRate",
    this->covariance_twist_yaw_rate_, 0.25);
  declare_param(this, "odom/covariance/recoveryScale",
    this->covariance_recovery_scale_, 10.0);
  declare_param(this, "odom/covariance/recoveryScans",
    this->covariance_recovery_scans_, 3);
  declare_param(this, "odom/publish/maxScanAge",
    this->odom_publish_max_scan_age_, 0.2);
  declare_param(this, "odom/recovery/enabled", this->recovery_enabled_, true);
  declare_param(this, "odom/recovery/rejectedScans",
    this->recovery_rejected_scans_, 20);
  declare_param(this, "odom/recovery/confirmationScans",
    this->recovery_confirmation_scans_, 3);

  const auto valid_diagonal = [](const std::vector<double>& diagonal) {
      return diagonal.size() == 6 &&
        std::all_of(diagonal.begin(), diagonal.end(), [](double value) {
          return std::isfinite(value) && value > 0.0;
        });
    };
  if (!valid_diagonal(this->pose_covariance_diagonal_) ||
      !valid_diagonal(this->twist_covariance_diagonal_)) {
    throw std::invalid_argument("DLIO covariance diagonals must contain six positive values");
  }
  if (!std::isfinite(this->imu_calib_time_) || this->imu_calib_time_ <= 0.0 ||
      this->imu_calibration_min_samples_ < 2 ||
      !std::isfinite(this->imu_calibration_gyro_stddev_max_) ||
      this->imu_calibration_gyro_stddev_max_ <= 0.0 ||
      !std::isfinite(this->imu_calibration_gyro_mean_max_) ||
      this->imu_calibration_gyro_mean_max_ <= 0.0 ||
      !std::isfinite(this->imu_calibration_accel_norm_stddev_max_) ||
      this->imu_calibration_accel_norm_stddev_max_ <= 0.0 ||
      !std::isfinite(this->imu_calibration_accel_gravity_tolerance_) ||
      this->imu_calibration_accel_gravity_tolerance_ <= 0.0 ||
      !std::isfinite(this->online_gyro_bias_stationary_time_) ||
      this->online_gyro_bias_stationary_time_ <= 0.0 ||
      !std::isfinite(this->online_gyro_bias_lidar_linear_max_) ||
      this->online_gyro_bias_lidar_linear_max_ <= 0.0 ||
      !std::isfinite(this->online_gyro_bias_lidar_angular_max_deg_) ||
      this->online_gyro_bias_lidar_angular_max_deg_ <= 0.0 ||
      !std::isfinite(this->online_gyro_bias_time_constant_) ||
      this->online_gyro_bias_time_constant_ <= 0.0 ||
      !std::isfinite(this->online_gyro_bias_max_delta_) ||
      this->online_gyro_bias_max_delta_ <= 0.0 ||
      !std::isfinite(this->imu_max_gap_seconds_) || this->imu_max_gap_seconds_ <= 0.0 ||
      !std::isfinite(this->imu_wait_timeout_seconds_) || this->imu_wait_timeout_seconds_ <= 0.0 ||
      !std::isfinite(this->gicp_max_correction_distance_) ||
      this->gicp_max_correction_distance_ <= 0.0 ||
      !std::isfinite(this->gicp_max_correction_rotation_deg_) ||
      this->gicp_max_correction_rotation_deg_ <= 0.0 ||
      !std::isfinite(this->gicp_min_correspondence_ratio_) ||
      this->gicp_min_correspondence_ratio_ <= 0.0 || this->gicp_min_correspondence_ratio_ > 1.0 ||
      this->gicp_quality_history_size_ < 1 || this->gicp_quality_warmup_scans_ < 1 ||
      !std::isfinite(this->gicp_max_normalized_error_factor_) ||
      this->gicp_max_normalized_error_factor_ <= 1.0 ||
      !std::isfinite(this->gicp_max_hessian_condition_) ||
      this->gicp_max_hessian_condition_ <= 1.0 ||
      !std::isfinite(this->gicp_max_hessian_condition_factor_) ||
      this->gicp_max_hessian_condition_factor_ <= 1.0 ||
      !std::isfinite(this->gicp_soft_correction_distance_) ||
      this->gicp_soft_correction_distance_ <= 0.0 ||
      !std::isfinite(this->gicp_soft_correction_rotation_deg_) ||
      this->gicp_soft_correction_rotation_deg_ <= 0.0 ||
      this->gicp_correction_confirmation_scans_ < 1 ||
      !std::isfinite(this->gicp_correction_consistency_distance_) ||
      this->gicp_correction_consistency_distance_ <= 0.0 ||
      !std::isfinite(this->gicp_correction_consistency_rotation_deg_) ||
      this->gicp_correction_consistency_rotation_deg_ <= 0.0 ||
      !std::isfinite(this->covariance_degraded_after_seconds_) ||
      this->covariance_degraded_after_seconds_ < 0.0 ||
      !std::isfinite(this->covariance_pose_position_rate_) ||
      this->covariance_pose_position_rate_ < 0.0 ||
      !std::isfinite(this->covariance_pose_yaw_rate_) ||
      this->covariance_pose_yaw_rate_ < 0.0 ||
      !std::isfinite(this->covariance_twist_linear_rate_) ||
      this->covariance_twist_linear_rate_ < 0.0 ||
      !std::isfinite(this->covariance_twist_yaw_rate_) ||
      this->covariance_twist_yaw_rate_ < 0.0 ||
      !std::isfinite(this->covariance_recovery_scale_) ||
      this->covariance_recovery_scale_ < 1.0 || this->covariance_recovery_scans_ < 0 ||
      !std::isfinite(this->odom_publish_max_scan_age_) ||
      this->odom_publish_max_scan_age_ <= 0.0 || this->recovery_rejected_scans_ < 1 ||
      this->recovery_confirmation_scans_ < 1) {
    throw std::invalid_argument("DLIO timing, correction, and covariance parameters are invalid");
  }

  // Geometric Observer
  declare_param(this, "odom/geo/Kp", this->geo_Kp_, 1.0);
  declare_param(this, "odom/geo/Kv", this->geo_Kv_, 1.0);
  declare_param(this, "odom/geo/Kq", this->geo_Kq_, 1.0);
  declare_param(this, "odom/geo/Kab", this->geo_Kab_, 1.0);
  declare_param(this, "odom/geo/Kgb", this->geo_Kgb_, 1.0);
  declare_param(this, "odom/geo/abias_max", this->geo_abias_max_, 1.0);
  declare_param(this, "odom/geo/gbias_max", this->geo_gbias_max_, 1.0);
}

void OdomNode::start() {

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void OdomNode::setOdometryCovariance(
    nav_msgs::msg::Odometry& message, double stamp_seconds, double scale) const {
  message.pose.covariance.fill(0.0);
  message.twist.covariance.fill(0.0);

  for (size_t index = 0; index < 6; ++index) {
    message.pose.covariance[index * 6 + index] =
      scale * this->pose_covariance_diagonal_[index];
    message.twist.covariance[index * 6 + index] =
      scale * this->twist_covariance_diagonal_[index];
  }

  const double last_accepted = this->last_accepted_scan_stamp_.load();
  const double degraded_age = std::max(
    0.0, stamp_seconds - last_accepted - this->covariance_degraded_after_seconds_);
  if (last_accepted <= 0.0 || degraded_age <= 0.0) {
    return;
  }

  const double pose_position_increase =
    this->covariance_pose_position_rate_ * degraded_age;
  const double pose_yaw_increase = this->covariance_pose_yaw_rate_ * degraded_age;
  const double twist_linear_increase =
    this->covariance_twist_linear_rate_ * degraded_age;
  const double twist_yaw_increase = this->covariance_twist_yaw_rate_ * degraded_age;

  message.pose.covariance[0] += pose_position_increase;
  message.pose.covariance[7] += pose_position_increase;
  message.pose.covariance[35] += pose_yaw_increase;
  message.twist.covariance[0] += twist_linear_increase;
  message.twist.covariance[7] += twist_linear_increase;
  message.twist.covariance[35] += twist_yaw_increase;
}

nav_msgs::msg::Odometry OdomNode::makeOdometryMessage(
    const State& state_snapshot, const rclcpp::Time& stamp, double covariance_scale) const {
  nav_msgs::msg::Odometry message;
  message.header.stamp = stamp;
  message.header.frame_id = this->odom_frame;
  message.child_frame_id = this->baselink_frame;

  message.pose.pose.position.x = state_snapshot.p[0];
  message.pose.pose.position.y = state_snapshot.p[1];
  message.pose.pose.position.z = state_snapshot.p[2];
  message.pose.pose.orientation.w = state_snapshot.q.w();
  message.pose.pose.orientation.x = state_snapshot.q.x();
  message.pose.pose.orientation.y = state_snapshot.q.y();
  message.pose.pose.orientation.z = state_snapshot.q.z();

  message.twist.twist.linear.x = state_snapshot.v.lin.w[0];
  message.twist.twist.linear.y = state_snapshot.v.lin.w[1];
  message.twist.twist.linear.z = state_snapshot.v.lin.w[2];
  message.twist.twist.angular.x = state_snapshot.v.ang.b[0];
  message.twist.twist.angular.y = state_snapshot.v.ang.b[1];
  message.twist.twist.angular.z = state_snapshot.v.ang.b[2];

  this->setOdometryCovariance(message, stamp.seconds(), covariance_scale);
  return message;
}

bool OdomNode::odometryOutputHealthy() const {
  if (!this->odom_ready_.load() || !this->imu_healthy_.load() ||
      !this->scan_healthy_.load()) {
    return false;
  }

  const int64_t accepted_ns = this->last_accepted_steady_ns_.load();
  if (accepted_ns <= 0) {
    return false;
  }
  const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  return static_cast<double>(now_ns - accepted_ns) * 1e-9 <=
    this->odom_publish_max_scan_age_;
}

void OdomNode::publishPose() {

  if (!this->odometryOutputHealthy()) {
    return;
  }

  const int64_t imu_stamp_ns = this->latest_imu_stamp_ns_.load();
  if (imu_stamp_ns <= 0) {
    return;
  }

  State state_snapshot;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    state_snapshot = this->state;
  }
  const rclcpp::Time publish_stamp(imu_stamp_ns);
  const double covariance_scale = this->recovery_scans_remaining_.load() > 0 ?
    this->covariance_recovery_scale_ : 1.0;

  this->odom_pub->publish(
    this->makeOdometryMessage(state_snapshot, publish_stamp, covariance_scale));

  // geometry_msgs::msg::PoseStamped
  this->pose_ros.header.stamp = publish_stamp;
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = state_snapshot.p[0];
  this->pose_ros.pose.position.y = state_snapshot.p[1];
  this->pose_ros.pose.position.z = state_snapshot.p[2];

  this->pose_ros.pose.orientation.w = state_snapshot.q.w();
  this->pose_ros.pose.orientation.x = state_snapshot.q.x();
  this->pose_ros.pose.orientation.y = state_snapshot.q.y();
  this->pose_ros.pose.orientation.z = state_snapshot.q.z();

  this->pose_pub->publish(this->pose_ros);

}

void OdomNode::publishScanOdometry() {
  if (!std::isfinite(this->scan_stamp) || this->scan_stamp <= 0.0 ||
      !this->odometryOutputHealthy()) {
    return;
  }

  State state_snapshot;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    state_snapshot = this->state;
  }

  int recovery_remaining = this->recovery_scans_remaining_.load();
  const double covariance_scale = recovery_remaining > 0 ?
    this->covariance_recovery_scale_ : 1.0;
  this->scan_odom_pub->publish(this->makeOdometryMessage(
    state_snapshot,
    rclcpp::Time(static_cast<int64_t>(std::llround(this->scan_stamp * 1.0e9))),
    covariance_scale));

  while (recovery_remaining > 0 &&
      !this->recovery_scans_remaining_.compare_exchange_weak(
        recovery_remaining, recovery_remaining - 1)) {}
}

void OdomNode::publishDiagnostics() {
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = this->now();

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "DLIO: Odometry Health";
  status.hardware_id = "livox_mid360";

  double accepted_age = std::numeric_limits<double>::infinity();
  const int64_t accepted_ns = this->last_accepted_steady_ns_.load();
  if (accepted_ns > 0) {
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    accepted_age = static_cast<double>(now_ns - accepted_ns) * 1e-9;
  }

  const bool calibrated = this->imu_calibrated.load();
  const bool healthy = this->odometryOutputHealthy();
  if (!calibrated) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = "waiting for stationary IMU calibration";
  } else if (!this->odom_ready_.load()) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = "waiting for the first accepted LiDAR scan";
  } else if (!healthy) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    if (!this->imu_healthy_.load() || !this->scan_healthy_.load()) {
      std::lock_guard<std::mutex> lock(this->health_mutex_);
      status.message = this->health_reason_;
    } else {
      std::ostringstream message;
      message << "no accepted LiDAR scan within "
              << this->odom_publish_max_scan_age_ << " s";
      status.message = message.str();
    }
  } else {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = "scan-corrected odometry is healthy";
  }

  const auto add_value = [&status](const std::string& key, const auto& value) {
      diagnostic_msgs::msg::KeyValue item;
      item.key = key;
      std::ostringstream stream;
      stream << value;
      item.value = stream.str();
      status.values.push_back(std::move(item));
    };

  Eigen::Vector3f gyro_bias;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    gyro_bias = this->state.b.gyro;
  }
  add_value("last_accepted_scan_age_s", accepted_age);
  add_value("consecutive_rejected_scans", this->consecutive_rejected_scans_.load());
  add_value("recovery_state", this->recoveryStateName());
  add_value("recovery_reseed_count", this->recovery_reseed_count_.load());
  add_value("recovery_confirmation_count", this->recovery_confirmation_count_.load());
  add_value("prediction_anchor_stamp", this->prediction_anchor_diagnostic_stamp_.load());
  add_value("imu_healthy", this->imu_healthy_.load());
  add_value("scan_healthy", this->scan_healthy_.load());
  add_value("correspondence_ratio", this->last_correspondence_ratio_.load());
  add_value("normalized_registration_error",
    this->last_normalized_registration_error_.load());
  add_value("hessian_condition", this->last_hessian_condition_.load());
  add_value("gyro_bias_x", gyro_bias.x());
  add_value("gyro_bias_y", gyro_bias.y());
  add_value("gyro_bias_z", gyro_bias.z());
  array.status.push_back(std::move(status));
  this->diagnostics_pub->publish(array);
}

void OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  const rclcpp::Time publish_stamp(this->latest_imu_stamp_ns_.load());
  State state_snapshot;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    state_snapshot = this->state;
  }

  // nav_msgs::msg::Path
  this->path_ros.header.stamp = publish_stamp;
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::msg::PoseStamped p;
  p.header.stamp = publish_stamp;
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = state_snapshot.p[0];
  p.pose.position.y = state_snapshot.p[1];
  p.pose.position.z = state_snapshot.p[2];
  p.pose.orientation.w = state_snapshot.q.w();
  p.pose.orientation.x = state_snapshot.q.x();
  p.pose.orientation.y = state_snapshot.q.y();
  p.pose.orientation.z = state_snapshot.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub->publish(this->path_ros);

  // transform: odom to baselink
  geometry_msgs::msg::TransformStamped transformStamped;

  if (this->publish_odom_tf_) {
    transformStamped.header.stamp = publish_stamp;
    transformStamped.header.frame_id = this->odom_frame;
    transformStamped.child_frame_id = this->baselink_frame;

    transformStamped.transform.translation.x = state_snapshot.p[0];
    transformStamped.transform.translation.y = state_snapshot.p[1];
    transformStamped.transform.translation.z = state_snapshot.p[2];

    transformStamped.transform.rotation.w = state_snapshot.q.w();
    transformStamped.transform.rotation.x = state_snapshot.q.x();
    transformStamped.transform.rotation.y = state_snapshot.q.y();
    transformStamped.transform.rotation.z = state_snapshot.q.z();

    br->sendTransform(transformStamped);
  }

  if (this->publish_sensor_tf_) {
    // transform: baselink to imu
    transformStamped.header.stamp = publish_stamp;
    transformStamped.header.frame_id = this->baselink_frame;
    transformStamped.child_frame_id = this->imu_frame;

    transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
    transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
    transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

    Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
    transformStamped.transform.rotation.w = q.w();
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();

    br->sendTransform(transformStamped);

    // transform: baselink to lidar
    transformStamped.header.stamp = publish_stamp;
    transformStamped.header.frame_id = this->baselink_frame;
    transformStamped.child_frame_id = this->lidar_frame;

    transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
    transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
    transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

    Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
    transformStamped.transform.rotation.w = qq.w();
    transformStamped.transform.rotation.x = qq.x();
    transformStamped.transform.rotation.y = qq.y();
    transformStamped.transform.rotation.z = qq.z();

    br->sendTransform(transformStamped);
  }

}

void OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ = std::make_shared<pcl::PointCloud<PointType>>();

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  sensor_msgs::msg::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  deskewed_ros.header.stamp = this->scan_header_stamp;
  deskewed_ros.header.frame_id = this->odom_frame;
  this->deskewed_pub->publish(deskewed_ros);

}

void OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, rclcpp::Time timestamp) {

  // Push back
  geometry_msgs::msg::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  this->kf_pose_ros.poses.push_back(p);

  // Publish
  this->kf_pose_ros.header.stamp = timestamp;
  this->kf_pose_ros.header.frame_id = this->odom_frame;
  this->kf_pose_pub->publish(this->kf_pose_ros);

  // publish keyframe scan for map
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub->publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::msg::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub->publish(keyframe_cloud_ros);
  }

}

void OdomNode::getScanFromROS(const sensor_msgs::msg::PointCloud2::SharedPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::fromROSMsg(*pc, *original_scan_);

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // automatically detect sensor type
  this->sensor = SensorType::UNKNOWN;
  
  // Guard: cloud may be empty after NaN removal / cropping
  if (original_scan_->points.empty()) {
    this->scan_header_stamp = pc->header.stamp;
    this->original_scan = original_scan_;
    return;
  }

  for (auto &field : pc->fields) {
    if (field.name == "t") {
      this->sensor = SensorType::OUSTER;
      break;
    } else if (field.name == "time") {
      this->sensor = SensorType::VELODYNE;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp < 1e14) {
      this->sensor = SensorType::HESAI;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp > 1e14) {
      this->sensor = SensorType::LIVOX;
      break;
    }
  }

  if (this->sensor == SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void OdomNode::setPredictionAnchor(
    const Eigen::Matrix4f& pose, const Eigen::Vector3f& velocity, double stamp) {
  Eigen::Quaternionf orientation(pose.block<3, 3>(0, 0));
  orientation.normalize();
  this->prediction_anchor_position_ = pose.block<3, 1>(0, 3);
  this->prediction_anchor_orientation_ = orientation;
  this->prediction_anchor_velocity_ = velocity;
  this->prediction_anchor_stamp_ = stamp;
  this->prediction_anchor_diagnostic_stamp_.store(stamp);
}

void OdomNode::resetPredictionAnchorToTrustedPose(double stamp) {
  this->setPredictionAnchor(this->T, Eigen::Vector3f::Zero(), stamp);
}

const char* OdomNode::recoveryStateName() const {
  switch (this->recovery_state_.load()) {
    case RecoveryState::kTracking:
      return "tracking";
    case RecoveryState::kReseedPending:
      return "reseed_pending";
    case RecoveryState::kConfirming:
      return "confirming";
  }
  return "unknown";
}

void OdomNode::scheduleRecovery(uint64_t rejected_count) {
  if (!this->recovery_enabled_ ||
      rejected_count < static_cast<uint64_t>(this->recovery_rejected_scans_) ||
      this->recovery_state_.load() == RecoveryState::kReseedPending) {
    return;
  }

  this->recovery_confirmation_count_.store(0);
  this->recovery_state_.store(RecoveryState::kReseedPending);
  {
    std::lock_guard<std::mutex> lock(this->health_mutex_);
    this->health_reason_ = "recovery pending after sustained LiDAR scan rejection";
  }
  RCLCPP_ERROR(
    this->get_logger(),
    "Scheduling DLIO local-map reseed after %llu consecutive rejected scans",
    static_cast<unsigned long long>(rejected_count));
}

bool OdomNode::submapReadyForRecovery() {
  if (!this->submap_future.valid()) {
    return true;
  }
  if (this->submap_future.wait_for(std::chrono::seconds(0)) !=
      std::future_status::ready) {
    return false;
  }
  try {
    this->submap_future.get();
  } catch (const std::exception& exception) {
    this->markOdometryUnhealthy(
      std::string("submap build failed before recovery: ") + exception.what());
    RCLCPP_ERROR(
      this->get_logger(), "Unable to begin DLIO recovery: %s", exception.what());
    return false;
  }
  return true;
}

void OdomNode::reseedLocalMap() {
  const Eigen::Matrix4f rebase = this->T * this->T_prior.inverse();
  auto rebased_deskewed = std::make_shared<pcl::PointCloud<PointType>>();
  auto rebased_current = std::make_shared<pcl::PointCloud<PointType>>();
  pcl::transformPointCloud(*this->deskewed_scan, *rebased_deskewed, rebase);
  pcl::transformPointCloud(*this->current_scan, *rebased_current, rebase);
  this->deskewed_scan = rebased_deskewed;
  this->current_scan = rebased_current;

  this->T_prior = this->T;
  this->T_corr = Eigen::Matrix4f::Identity();
  this->propagateGICP();
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    this->state.p = this->lidarPose.p;
    this->state.q = this->lidarPose.q;
    this->state.v.lin.b.setZero();
    this->state.v.lin.w.setZero();
    this->state.v.ang.b.setZero();
    this->state.v.ang.w.setZero();
    this->geo.prev_p = this->state.p;
    this->geo.prev_q = this->state.q;
    this->geo.prev_vel.setZero();
  }
  this->setPredictionAnchor(this->T, Eigen::Vector3f::Zero(), this->scan_stamp);
  this->prev_scan_stamp = this->scan_stamp;

  {
    std::lock_guard<std::mutex> lock(this->keyframes_mutex);
    this->keyframes.clear();
    this->keyframe_timestamps.clear();
    this->keyframe_normals.clear();
    this->keyframe_transformations.clear();
  }
  this->num_processed_keyframes = 0;
  this->keyframe_convex.clear();
  this->keyframe_concave.clear();
  this->submap_kf_idx_curr.clear();
  this->submap_kf_idx_prev.clear();
  this->submap_cloud = std::make_shared<const pcl::PointCloud<PointType>>();
  this->submap_normals.reset();
  this->submap_kdtree.reset();
  this->submap_hasChanged = true;
  this->new_submap_is_ready = false;

  this->registration_error_history_.clear();
  this->registration_condition_history_.clear();
  this->pending_correction_valid_ = false;
  this->pending_correction_count_ = 0;
  this->online_bias_samples_.clear();
  this->consecutive_rejected_scans_.store(0);
  this->recovery_confirmation_count_.store(0);
  this->recovery_state_.store(RecoveryState::kConfirming);
  this->recovery_reseed_count_.fetch_add(1);
  this->markOdometryUnhealthy("local map reseeded; confirming LiDAR registration");
  RCLCPP_WARN(
    this->get_logger(),
    "Reseeded DLIO local map at the last trusted pose; waiting for %d consistent scans",
    this->recovery_confirmation_scans_);
}

bool OdomNode::recoveryRegistrationConfirmed() {
  const int confirmed = this->recovery_confirmation_count_.fetch_add(1) + 1;
  if (confirmed < this->recovery_confirmation_scans_) {
    std::ostringstream reason;
    reason << "confirming LiDAR registration after local-map reseed (" << confirmed
           << "/" << this->recovery_confirmation_scans_ << ")";
    this->markOdometryUnhealthy(reason.str());
    return false;
  }

  this->recovery_state_.store(RecoveryState::kTracking);
  RCLCPP_INFO(
    this->get_logger(), "DLIO recovery confirmed after %d consistent scans", confirmed);
  return true;
}

void OdomNode::rejectScan(const ImuIntegrationResult& result, bool reanchor) {
  const uint64_t rejected_count = this->consecutive_rejected_scans_.fetch_add(1) + 1;
  this->markOdometryUnhealthy(result.reason);
  this->recovery_confirmation_count_.store(0);
  this->pending_correction_valid_ = false;
  this->pending_correction_count_ = 0;
  if (reanchor) {
    // Only advance the temporal anchor. The rejected scan must never enter GICP or the map.
    this->prev_scan_stamp = this->scan_stamp;
    this->resetPredictionAnchorToTrustedPose(this->scan_stamp);
  }
  this->scheduleRecovery(rejected_count);

  RCLCPP_ERROR_THROTTLE(
    this->get_logger(), *this->get_clock(), 1000,
    "Rejecting LiDAR scan at %.6f: %s (IMU range %.6f to %.6f, consecutive rejects: %llu)",
    this->scan_stamp, result.reason.c_str(), result.oldest_imu_stamp,
    result.newest_imu_stamp, static_cast<unsigned long long>(rejected_count));
}

void OdomNode::rejectScan(
    const std::string& reason, bool preserve_pending_correction) {
  const uint64_t rejected_count = this->consecutive_rejected_scans_.fetch_add(1) + 1;
  this->markOdometryUnhealthy(reason);
  this->recovery_confirmation_count_.store(0);
  if (!preserve_pending_correction) {
    this->pending_correction_valid_ = false;
    this->pending_correction_count_ = 0;
  }
  this->scheduleRecovery(rejected_count);
  RCLCPP_ERROR_THROTTLE(
    this->get_logger(), *this->get_clock(), 1000,
    "Rejecting LiDAR scan at %.6f: %s (consecutive rejects: %llu)",
    this->scan_stamp, reason.c_str(), static_cast<unsigned long long>(rejected_count));
}

void OdomNode::markOdometryUnhealthy(const std::string& reason) {
  this->scan_healthy_.store(false);
  this->recovery_scans_remaining_.store(this->covariance_recovery_scans_);
  std::lock_guard<std::mutex> lock(this->health_mutex_);
  this->health_reason_ = reason;
}

void OdomNode::markScanAccepted() {
  const uint64_t rejected_count = this->consecutive_rejected_scans_.exchange(0);
  this->last_accepted_scan_stamp_.store(this->scan_stamp);
  this->last_accepted_steady_ns_.store(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  this->odom_ready_.store(true);
  this->imu_healthy_.store(true);
  this->scan_healthy_.store(true);
  this->recovery_state_.store(RecoveryState::kTracking);
  this->recovery_confirmation_count_.store(0);
  this->updateAcceptedLidarMotion();
  {
    std::lock_guard<std::mutex> lock(this->health_mutex_);
    this->health_reason_ = "scan-corrected odometry is healthy";
  }
  if (rejected_count > 0) {
    RCLCPP_INFO(
      this->get_logger(), "LiDAR odometry recovered after rejecting %llu scan(s)",
      static_cast<unsigned long long>(rejected_count));
  }
}

bool OdomNode::preprocessPoints() {

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    if (!this->deskewPointcloud()) {
      return false;
    }

  } else {

    this->scan_stamp = rclcpp::Time(this->scan_header_stamp).seconds();

    if (!std::isfinite(this->scan_stamp)) {
      this->rejectScan("non-finite LiDAR timestamp");
      return false;
    }

    if (this->temporal_reanchor_pending_.exchange(false)) {
      this->prev_scan_stamp = this->scan_stamp;
      this->resetPredictionAnchorToTrustedPose(this->scan_stamp);
      this->rejectScan("re-anchoring after an IMU timestamp discontinuity");
      return false;
    }

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      std::lock_guard<std::mutex> imu_lock(this->mtx_imu);
      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return false;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan
      this->setPredictionAnchor(
        this->T_prior, Eigen::Vector3f::Zero(), this->scan_stamp);

    } else {

      // IMU prior for second scan onwards
      const ImuIntegrationResult integration = this->integrateImu(
        this->prediction_anchor_stamp_, this->prediction_anchor_orientation_,
        this->prediction_anchor_position_, this->prediction_anchor_velocity_,
        {this->scan_stamp});
      if (!integration.ok()) {
        const bool reanchor =
          integration.status == ImuIntegrationStatus::kHistoryUnavailable ||
          integration.status == ImuIntegrationStatus::kImuGap ||
          integration.status == ImuIntegrationStatus::kIncomplete;
        this->rejectScan(integration, reanchor);
        return false;
      }
      this->T_prior = integration.frames.back();
      this->setPredictionAnchor(
        this->T_prior, integration.anchor_velocity, this->scan_stamp);

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_ = std::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan);
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

  return true;

}

bool OdomNode::deskewPointcloud() {
  if (!this->original_scan || this->original_scan->points.empty()) {
    this->deskewed_scan.reset(new pcl::PointCloud<PointType>());
    this->deskew_status = false;
    this->rejectScan("empty LiDAR point cloud");
    return false;
  }
  pcl::PointCloud<PointType>::Ptr deskewed_scan_ = std::make_shared<pcl::PointCloud<PointType>>(1, this->original_scan->points.size());
  // deskewed_scan_->points.resize(this->original_scan->points.size());
  // individual point timestamps should be relative to this time
  double sweep_ref_time = rclcpp::Time(this->scan_header_stamp).seconds();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };
  } else if (this->sensor == SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  if (!std::isfinite(this->scan_stamp) ||
      !std::all_of(timestamps.begin(), timestamps.end(), [](double stamp) {
        return std::isfinite(stamp);
      })) {
    this->deskew_status = false;
    this->rejectScan("non-finite point timestamp in LiDAR scan");
    return false;
  }

  if (this->temporal_reanchor_pending_.exchange(false)) {
    this->prev_scan_stamp = this->scan_stamp;
    this->resetPredictionAnchorToTrustedPose(this->scan_stamp);
    this->deskew_status = false;
    this->rejectScan("re-anchoring after an IMU timestamp discontinuity");
    return false;
  }

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    std::lock_guard<std::mutex> imu_lock(this->mtx_imu);
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return false;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    this->setPredictionAnchor(
      this->T_prior, Eigen::Vector3f::Zero(), this->scan_stamp);
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return true;
  }

  // IMU prior & deskewing for second scan onwards
  const ImuIntegrationResult integration = this->integrateImu(
    this->prediction_anchor_stamp_, this->prediction_anchor_orientation_,
    this->prediction_anchor_position_, this->prediction_anchor_velocity_, timestamps);
  this->deskew_size = integration.frames.size();
  if (!integration.ok()) {
    const bool reanchor =
      integration.status == ImuIntegrationStatus::kHistoryUnavailable ||
      integration.status == ImuIntegrationStatus::kImuGap ||
      integration.status == ImuIntegrationStatus::kIncomplete;
    this->deskew_status = false;
    this->rejectScan(integration, reanchor);
    return false;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = integration.frames[median_pt_index];
  this->setPredictionAnchor(
    this->T_prior, integration.anchor_velocity, this->scan_stamp);

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = integration.frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;
  return true;

}

void OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);

}

void OdomNode::setInputSource() {
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}

void OdomNode::callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pc) {

  {
    std::lock_guard<std::mutex> lock(this->main_loop_running_mutex);
    this->main_loop_running = true;
  }
  ScopeExit finish_main_loop([this]() {
    {
      std::lock_guard<std::mutex> lock(this->main_loop_running_mutex);
      this->main_loop_running = false;
    }
    this->submap_build_cv.notify_one();
  });

  double then = this->now().seconds();

  if (this->recovery_state_.load() == RecoveryState::kReseedPending &&
      !this->submapReadyForRecovery()) {
    return;
  }

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = rclcpp::Time(pc->header.stamp).seconds();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  if (!this->preprocessPoints()) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    this->rejectScan("too few points after LiDAR preprocessing");
    return;
  }

  if (this->recovery_state_.load() == RecoveryState::kReseedPending) {
    this->reseedLocalMap();
  }

  // Compute Metrics
  this->computeMetrics();

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    if (this->recovery_state_.load() != RecoveryState::kConfirming) {
      this->markScanAccepted();
      this->publishScanOdometry();
    }
    finish_main_loop.run();
    this->submap_future =
      std::async( std::launch::async, &OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  if (!this->getNextPose()) {
    return;
  }
  if (this->recovery_state_.load() == RecoveryState::kConfirming &&
      !this->recoveryRegistrationConfirmed()) {
    this->prev_scan_stamp = this->scan_stamp;
    return;
  }
  this->markScanAccepted();

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    finish_main_loop.run();
    this->submap_future =
      std::async( std::launch::async, &OdomNode::buildKeyframesAndSubmap, this, this->state );
  }

  // Update trajectory
  this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );

  // Update time stamps
  this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  this->publishToROS(published_cloud, this->T_corr);
  this->publishScanOdometry();

  // Update some statistics
  this->comp_times.push_back(this->now().seconds() - then);
  this->gicp_hasConverged = true;

  // Read this parameter on each scan so the dashboard can be toggled at runtime.
  this->get_parameter("debug/enable", this->debug_enabled_);
  if (this->debug_enabled_) {
    this->debug();
  }

  this->geo.first_opt_done = true;

}

bool OdomNode::updateImuCalibration(
    double stamp, const Eigen::Vector3f& ang_vel, const Eigen::Vector3f& lin_accel) {
  this->imu_calibration_samples_.push_back({stamp, ang_vel, lin_accel});

  if (this->imu_calibration_samples_.size() <
        static_cast<size_t>(this->imu_calibration_min_samples_) ||
      this->imu_calibration_samples_.back().stamp -
        this->imu_calibration_samples_.front().stamp < this->imu_calib_time_) {
    return false;
  }

  Eigen::Vector3f gyro_mean = Eigen::Vector3f::Zero();
  Eigen::Vector3f accel_mean = Eigen::Vector3f::Zero();
  double accel_norm_mean = 0.0;
  for (const auto& sample : this->imu_calibration_samples_) {
    gyro_mean += sample.ang_vel;
    accel_mean += sample.lin_accel;
    accel_norm_mean += sample.lin_accel.norm();
  }
  const double count = static_cast<double>(this->imu_calibration_samples_.size());
  gyro_mean /= count;
  accel_mean /= count;
  accel_norm_mean /= count;

  Eigen::Vector3f gyro_variance = Eigen::Vector3f::Zero();
  double accel_norm_variance = 0.0;
  for (const auto& sample : this->imu_calibration_samples_) {
    const Eigen::Vector3f gyro_error = sample.ang_vel - gyro_mean;
    gyro_variance += gyro_error.cwiseProduct(gyro_error);
    const double accel_norm_error = sample.lin_accel.norm() - accel_norm_mean;
    accel_norm_variance += accel_norm_error * accel_norm_error;
  }
  const Eigen::Vector3f gyro_stddev = (gyro_variance / count).cwiseSqrt();
  const double accel_norm_stddev = std::sqrt(accel_norm_variance / count);

  const bool stationary =
    gyro_stddev.maxCoeff() <= this->imu_calibration_gyro_stddev_max_ &&
    gyro_mean.norm() <= this->imu_calibration_gyro_mean_max_ &&
    accel_norm_stddev <= this->imu_calibration_accel_norm_stddev_max_ &&
    std::abs(accel_norm_mean - std::abs(this->gravity_)) <=
      this->imu_calibration_accel_gravity_tolerance_;
  if (!stationary) {
    const ImuCalibrationSample current = this->imu_calibration_samples_.back();
    this->imu_calibration_samples_.clear();
    this->imu_calibration_samples_.push_back(current);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "IMU calibration window was not stationary; restarting collection "
      "(gyro std max %.6f rad/s, gyro mean norm %.6f rad/s, accel norm std %.6f m/s^2)",
      gyro_stddev.maxCoeff(), gyro_mean.norm(), accel_norm_stddev);
    return false;
  }

  Eigen::Vector3f grav_vec(0.0f, 0.0f, static_cast<float>(this->gravity_));
  std::lock_guard<std::mutex> state_lock(this->geo.mtx);
  if (this->gravity_align_) {
    grav_vec = (accel_mean - this->state.b.accel).normalized() * std::abs(this->gravity_);
    const Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(
      grav_vec, Eigen::Vector3f(0.0f, 0.0f, static_cast<float>(this->gravity_)));
    this->state.q = grav_q;
    this->T.block(0, 0, 3, 3) = grav_q.toRotationMatrix();
    this->lidarPose.q = grav_q;
  }
  if (this->calibrate_accel_) {
    this->state.b.accel = accel_mean - grav_vec;
  }
  if (this->calibrate_gyro_) {
    this->state.b.gyro = gyro_mean;
  }
  this->startup_gyro_bias_ = this->state.b.gyro;

  RCLCPP_INFO(
    this->get_logger(),
    "Stationary IMU calibration complete with %zu samples; gyro bias "
    "[%.8f, %.8f, %.8f] rad/s",
    this->imu_calibration_samples_.size(), this->state.b.gyro.x(),
    this->state.b.gyro.y(), this->state.b.gyro.z());
  this->imu_calibration_samples_.clear();
  this->imu_calibrated.store(true);
  this->prev_imu_stamp = stamp;
  return true;
}

void OdomNode::updateAcceptedLidarMotion() {
  if (!this->accepted_lidar_pose_available_) {
    this->previous_accepted_lidar_position_ = this->lidarPose.p;
    this->previous_accepted_lidar_orientation_ = this->lidarPose.q;
    this->previous_accepted_lidar_stamp_ = this->scan_stamp;
    this->accepted_lidar_pose_available_ = true;
    return;
  }

  const double dt = this->scan_stamp - this->previous_accepted_lidar_stamp_;
  if (std::isfinite(dt) && dt > 0.0) {
    const double linear_speed =
      (this->lidarPose.p - this->previous_accepted_lidar_position_).norm() / dt;
    Eigen::Quaternionf delta =
      this->previous_accepted_lidar_orientation_.conjugate() * this->lidarPose.q;
    delta.normalize();
    const double angular_distance = 2.0 * std::acos(std::clamp(
      std::abs(static_cast<double>(delta.w())), 0.0, 1.0));
    this->accepted_lidar_linear_speed_.store(linear_speed);
    this->accepted_lidar_angular_speed_.store(angular_distance / dt);
  }
  this->previous_accepted_lidar_position_ = this->lidarPose.p;
  this->previous_accepted_lidar_orientation_ = this->lidarPose.q;
  this->previous_accepted_lidar_stamp_ = this->scan_stamp;
}

void OdomNode::updateOnlineGyroBias(
    double stamp, double dt, const Eigen::Vector3f& ang_vel,
    const Eigen::Vector3f& lin_accel) {
  if (!this->online_gyro_bias_enabled_ || !this->odometryOutputHealthy() ||
      this->accepted_lidar_linear_speed_.load() >
        this->online_gyro_bias_lidar_linear_max_ ||
      this->accepted_lidar_angular_speed_.load() >
        this->online_gyro_bias_lidar_angular_max_deg_ * M_PI / 180.0) {
    this->online_bias_samples_.clear();
    return;
  }

  this->online_bias_samples_.push_back({stamp, ang_vel, lin_accel});
  while (this->online_bias_samples_.size() > 1 &&
      stamp - this->online_bias_samples_[1].stamp >=
        this->online_gyro_bias_stationary_time_) {
    this->online_bias_samples_.pop_front();
  }
  if (this->online_bias_samples_.size() < 2 ||
      this->online_bias_samples_.back().stamp -
        this->online_bias_samples_.front().stamp <
          this->online_gyro_bias_stationary_time_) {
    return;
  }

  Eigen::Vector3f gyro_mean = Eigen::Vector3f::Zero();
  double accel_norm_mean = 0.0;
  for (const auto& sample : this->online_bias_samples_) {
    gyro_mean += sample.ang_vel;
    accel_norm_mean += sample.lin_accel.norm();
  }
  const double count = static_cast<double>(this->online_bias_samples_.size());
  gyro_mean /= count;
  accel_norm_mean /= count;

  Eigen::Vector3f gyro_variance = Eigen::Vector3f::Zero();
  double accel_norm_variance = 0.0;
  for (const auto& sample : this->online_bias_samples_) {
    const Eigen::Vector3f gyro_error = sample.ang_vel - gyro_mean;
    gyro_variance += gyro_error.cwiseProduct(gyro_error);
    const double accel_error = sample.lin_accel.norm() - accel_norm_mean;
    accel_norm_variance += accel_error * accel_error;
  }
  const Eigen::Vector3f gyro_stddev = (gyro_variance / count).cwiseSqrt();
  const double accel_norm_stddev = std::sqrt(accel_norm_variance / count);
  if (gyro_stddev.maxCoeff() > this->imu_calibration_gyro_stddev_max_ ||
      accel_norm_stddev > this->imu_calibration_accel_norm_stddev_max_) {
    return;
  }

  const Eigen::Vector3f min_bias = this->startup_gyro_bias_.array() -
    this->online_gyro_bias_max_delta_;
  const Eigen::Vector3f max_bias = this->startup_gyro_bias_.array() +
    this->online_gyro_bias_max_delta_;
  const Eigen::Vector3f target = gyro_mean.cwiseMax(min_bias).cwiseMin(max_bias);
  const double alpha = std::clamp(
    1.0 - std::exp(-dt / this->online_gyro_bias_time_constant_), 0.0, 1.0);
  std::lock_guard<std::mutex> state_lock(this->geo.mtx);
  this->state.b.gyro =
    ((1.0 - alpha) * this->state.b.gyro + alpha * target).eval();
}

void OdomNode::callbackImu(const sensor_msgs::msg::Imu::SharedPtr imu_raw) {

  const double imu_stamp_secs = rclcpp::Time(imu_raw->header.stamp).seconds();
  if (!std::isfinite(imu_stamp_secs)) {
    this->imu_healthy_.store(false);
    this->markOdometryUnhealthy("non-finite IMU timestamp");
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "Dropping IMU measurement with a non-finite timestamp");
    return;
  }

  double dt = 1.0 / 200.0;
  if (this->last_received_imu_stamp_ > 0.0) {
    dt = imu_stamp_secs - this->last_received_imu_stamp_;
    if (!std::isfinite(dt) || dt <= 0.0) {
      this->imu_healthy_.store(false);
      this->markOdometryUnhealthy("non-monotonic IMU timestamp");
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "Dropping non-monotonic IMU timestamp %.6f after %.6f (dt=%.9f)",
        imu_stamp_secs, this->last_received_imu_stamp_, dt);
      return;
    }
  }
  this->last_received_imu_stamp_ = imu_stamp_secs;
  this->first_imu_received = true;

  sensor_msgs::msg::Imu::SharedPtr imu = this->transformImu(imu_raw, dt);
  this->latest_imu_stamp_ns_.store(rclcpp::Time(imu->header.stamp).nanoseconds());

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (!this->imu_calibrated.load()) {
    this->updateImuCalibration(imu_stamp_secs, ang_vel, lin_accel);
    return;
  }

  if (this->prev_imu_stamp > 0.0) {
    dt = imu_stamp_secs - this->prev_imu_stamp;
  }
  if (!std::isfinite(dt) || dt <= 0.0) {
    this->imu_healthy_.store(false);
    this->markOdometryUnhealthy("invalid calibrated IMU interval");
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "Dropping invalid calibrated IMU interval %.9f seconds", dt);
    this->prev_imu_stamp = imu_stamp_secs;
    return;
  }
  this->imu_rates.push_back(1.0 / dt);

  // Store transformed raw measurements. Biases are applied when propagating or
  // to one copied deskew interval, so every interval uses a consistent snapshot.
  this->imu_meas.stamp = imu_stamp_secs;
  this->imu_meas.dt = dt;
  this->imu_meas.lin_accel = this->imu_accel_sm_ * lin_accel;
  this->imu_meas.ang_vel = ang_vel;
  this->prev_imu_stamp = this->imu_meas.stamp;

  if (dt > this->imu_max_gap_seconds_) {
    {
      std::lock_guard<std::mutex> lock(this->mtx_imu);
      this->imu_buffer.clear();
      this->imu_buffer.push_front(this->imu_meas);
    }
    this->imu_healthy_.store(false);
    this->markOdometryUnhealthy("IMU timestamp gap; waiting for LiDAR re-anchor");
    this->temporal_reanchor_pending_.store(true);
    this->online_bias_samples_.clear();
    this->cv_imu_stamp.notify_one();
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "IMU gap of %.6f seconds exceeds %.6f seconds; holding state and re-anchoring LiDAR",
      dt, this->imu_max_gap_seconds_);
    return;
  }

  this->updateOnlineGyroBias(
    imu_stamp_secs, dt, this->imu_meas.ang_vel, this->imu_meas.lin_accel);

  {
    std::lock_guard<std::mutex> lock(this->mtx_imu);
    this->imu_buffer.push_front(this->imu_meas);
  }
  this->cv_imu_stamp.notify_one();

  if (this->geo.first_opt_done) {
    this->propagateState();
  }

}

bool OdomNode::validateRegistrationQuality(std::string& reason) {
  const double source_points = static_cast<double>(this->current_scan->size());
  const double correspondence_ratio = source_points > 0.0 ?
    static_cast<double>(this->gicp.num_correspondences) / source_points : 0.0;
  const double normalized_error = this->gicp.num_correspondences > 0 ?
    this->gicp.getFinalError() /
      static_cast<double>(this->gicp.num_correspondences) :
    std::numeric_limits<double>::infinity();

  const Eigen::Matrix<double, 6, 6> hessian = this->gicp.getFinalHessian();
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
    hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto singular_values = svd.singularValues().cwiseAbs();
  const double minimum_singular = singular_values.minCoeff();
  const double maximum_singular = singular_values.maxCoeff();
  const double hessian_condition = minimum_singular > 1.0e-12 ?
    maximum_singular / minimum_singular : std::numeric_limits<double>::infinity();

  this->last_correspondence_ratio_.store(correspondence_ratio);
  this->last_normalized_registration_error_.store(normalized_error);
  this->last_hessian_condition_.store(hessian_condition);

  if (!std::isfinite(correspondence_ratio) ||
      correspondence_ratio < this->gicp_min_correspondence_ratio_) {
    std::ostringstream stream;
    stream << "GICP correspondence ratio " << correspondence_ratio << " is below "
           << this->gicp_min_correspondence_ratio_;
    reason = stream.str();
    return false;
  }
  if (!std::isfinite(normalized_error) || normalized_error < 0.0) {
    reason = "GICP returned a non-finite normalized error";
    return false;
  }
  if (!std::isfinite(hessian_condition) ||
      hessian_condition > this->gicp_max_hessian_condition_) {
    std::ostringstream stream;
    stream << "GICP Hessian condition " << hessian_condition << " exceeds "
           << this->gicp_max_hessian_condition_;
    reason = stream.str();
    return false;
  }

  const auto median = [](const std::deque<double>& values) {
      std::vector<double> sorted(values.begin(), values.end());
      const size_t middle = sorted.size() / 2;
      std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
      double result = sorted[middle];
      if (sorted.size() % 2 == 0) {
        const auto lower = std::max_element(sorted.begin(), sorted.begin() + middle);
        result = 0.5 * (result + *lower);
      }
      return result;
    };

  if (this->registration_error_history_.size() >=
      static_cast<size_t>(this->gicp_quality_warmup_scans_)) {
    const double error_limit =
      median(this->registration_error_history_) *
      this->gicp_max_normalized_error_factor_;
    if (normalized_error > error_limit) {
      std::ostringstream stream;
      stream << "GICP normalized error " << normalized_error
             << " exceeds adaptive limit " << error_limit;
      reason = stream.str();
      return false;
    }

    const double condition_limit = std::min(
      this->gicp_max_hessian_condition_,
      median(this->registration_condition_history_) *
        this->gicp_max_hessian_condition_factor_);
    if (hessian_condition > condition_limit) {
      std::ostringstream stream;
      stream << "GICP Hessian condition " << hessian_condition
             << " exceeds adaptive limit " << condition_limit;
      reason = stream.str();
      return false;
    }
  }
  return true;
}

bool OdomNode::correctionNeedsConfirmation(
    const Eigen::Matrix4f& correction, double correction_distance,
    double correction_rotation_deg, std::string& reason) {
  if (this->gicp_correction_confirmation_scans_ <= 1 ||
      (correction_distance <= this->gicp_soft_correction_distance_ &&
       correction_rotation_deg <= this->gicp_soft_correction_rotation_deg_)) {
    this->pending_correction_valid_ = false;
    this->pending_correction_count_ = 0;
    return false;
  }

  if (!this->pending_correction_valid_) {
    this->pending_correction_ = correction;
    this->pending_correction_count_ = 1;
    this->pending_correction_valid_ = true;
  } else {
    const Eigen::Matrix4f delta = correction * this->pending_correction_.inverse();
    const double distance = delta.block<3, 1>(0, 3).norm();
    Eigen::Quaternionf rotation(delta.block<3, 3>(0, 0));
    rotation.normalize();
    const double rotation_deg =
      Eigen::AngleAxisf(rotation).angle() * 180.0 / M_PI;
    if (std::isfinite(distance) && std::isfinite(rotation_deg) &&
        distance <= this->gicp_correction_consistency_distance_ &&
        rotation_deg <= this->gicp_correction_consistency_rotation_deg_) {
      ++this->pending_correction_count_;
    } else {
      this->pending_correction_ = correction;
      this->pending_correction_count_ = 1;
    }
  }

  if (this->pending_correction_count_ >=
      this->gicp_correction_confirmation_scans_) {
    this->pending_correction_valid_ = false;
    this->pending_correction_count_ = 0;
    return false;
  }

  std::ostringstream stream;
  stream << "GICP correction pending confirmation (" << correction_distance
         << " m, " << correction_rotation_deg << " deg, candidate "
         << this->pending_correction_count_ << "/"
         << this->gicp_correction_confirmation_scans_ << ")";
  reason = stream.str();
  return true;
}

void OdomNode::recordAcceptedRegistrationQuality() {
  this->registration_error_history_.push_back(
    this->last_normalized_registration_error_.load());
  this->registration_condition_history_.push_back(
    this->last_hessian_condition_.load());
  while (this->registration_error_history_.size() >
      static_cast<size_t>(this->gicp_quality_history_size_)) {
    this->registration_error_history_.pop_front();
  }
  while (this->registration_condition_history_.size() >
      static_cast<size_t>(this->gicp_quality_history_size_)) {
    this->registration_condition_history_.pop_front();
  }
}

bool OdomNode::getNextPose() {

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    this->gicp.registerInputTarget(this->submap_cloud);

    // Set submap kdtree
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned = std::make_shared<pcl::PointCloud<PointType>>();
  this->gicp.align(*aligned);

  this->gicp_hasConverged = this->gicp.hasConverged();
  if (!this->gicp_hasConverged.load()) {
    this->pending_correction_valid_ = false;
    this->rejectScan("GICP did not converge");
    return false;
  }

  std::string quality_reason;
  if (!this->validateRegistrationQuality(quality_reason)) {
    this->pending_correction_valid_ = false;
    this->pending_correction_count_ = 0;
    this->rejectScan(quality_reason);
    return false;
  }

  // Both clouds are already in the global frame, so this matrix is the correction
  // applied to the IMU prior and should remain close to identity.
  const Eigen::Matrix4f correction = this->gicp.getFinalTransformation();
  if (!correction.allFinite()) {
    this->rejectScan("GICP returned a non-finite correction");
    return false;
  }

  const Eigen::Matrix3f correction_rotation = correction.block<3, 3>(0, 0);
  const float rotation_determinant = correction_rotation.determinant();
  Eigen::Quaternionf correction_quaternion(correction_rotation);
  if (!std::isfinite(rotation_determinant) || std::abs(rotation_determinant - 1.0f) > 0.1f ||
      !correction_quaternion.coeffs().allFinite() || correction_quaternion.norm() < 1e-6f) {
    this->rejectScan("GICP returned an invalid rotation");
    return false;
  }
  correction_quaternion.normalize();

  const double correction_distance = correction.block<3, 1>(0, 3).norm();
  const double correction_rotation_deg =
    Eigen::AngleAxisf(correction_quaternion).angle() * 180.0 / M_PI;
  if (!std::isfinite(correction_distance) || !std::isfinite(correction_rotation_deg) ||
      correction_distance > this->gicp_max_correction_distance_ ||
      correction_rotation_deg > this->gicp_max_correction_rotation_deg_) {
    std::ostringstream reason;
    reason << "GICP correction exceeds safety bounds (" << correction_distance << " m, "
           << correction_rotation_deg << " deg)";
    this->rejectScan(reason.str());
    return false;
  }

  std::string confirmation_reason;
  if (this->correctionNeedsConfirmation(
      correction, correction_distance, correction_rotation_deg,
      confirmation_reason)) {
    this->rejectScan(confirmation_reason, true);
    return false;
  }

  // Commit the correction only after all validation has passed.
  this->T_corr = correction;
  this->T = this->T_corr * this->T_prior;
  if (!this->T.allFinite()) {
    this->rejectScan("corrected pose is non-finite");
    return false;
  }

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  // Geometric observer update
  this->updateState();
  this->recordAcceptedRegistrationQuality();

  Eigen::Vector3f corrected_velocity;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    corrected_velocity = this->state.v.lin.w;
  }
  this->setPredictionAnchor(this->T, corrected_velocity, this->scan_stamp);

  return true;
}

bool OdomNode::imuMeasFromTimeRange(
    boost::circular_buffer<ImuMeas>& imu_snapshot, double start_time, double end_time,
    boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
    boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {
  if (imu_snapshot.size() < 2) {
    return false;
  }

  auto imu_it = imu_snapshot.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != imu_snapshot.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != imu_snapshot.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == imu_snapshot.end()) {
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

OdomNode::ImuIntegrationResult
OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  ImuIntegrationResult result;

  if (!std::isfinite(start_time) || sorted_timestamps.empty() ||
      !std::is_sorted(sorted_timestamps.begin(), sorted_timestamps.end()) ||
      start_time > sorted_timestamps.front() ||
      !std::all_of(sorted_timestamps.begin(), sorted_timestamps.end(), [](double stamp) {
        return std::isfinite(stamp);
      })) {
    result.status = ImuIntegrationStatus::kInvalidRange;
    result.reason = "invalid or unsorted integration timestamps";
    return result;
  }

  boost::circular_buffer<ImuMeas> imu_snapshot;
  {
    std::unique_lock<std::mutex> lock(this->mtx_imu);
    const double end_time = sorted_timestamps.back();
    const bool imu_arrived = this->cv_imu_stamp.wait_for(
      lock, std::chrono::duration<double>(this->imu_wait_timeout_seconds_),
      [this, end_time]() {
        return !this->imu_buffer.empty() && this->imu_buffer.front().stamp >= end_time;
      });
    if (!imu_arrived) {
      result.status = ImuIntegrationStatus::kImuTimeout;
      result.reason = "timed out waiting for IMU data through the end of the LiDAR scan";
      if (!this->imu_buffer.empty()) {
        result.oldest_imu_stamp = this->imu_buffer.back().stamp;
        result.newest_imu_stamp = this->imu_buffer.front().stamp;
      }
      return result;
    }

    result.oldest_imu_stamp = this->imu_buffer.back().stamp;
    result.newest_imu_stamp = this->imu_buffer.front().stamp;
    if (this->imu_buffer.size() < 2 || result.oldest_imu_stamp >= start_time) {
      result.status = ImuIntegrationStatus::kHistoryUnavailable;
      result.reason = "IMU buffer does not contain a sample before the integration start";
      return result;
    }
    imu_snapshot = this->imu_buffer;
  }

  Eigen::Vector3f gyro_bias;
  Eigen::Vector3f accel_bias;
  {
    std::lock_guard<std::mutex> state_lock(this->geo.mtx);
    gyro_bias = this->state.b.gyro;
    accel_bias = this->state.b.accel;
  }
  for (auto& measurement : imu_snapshot) {
    measurement.ang_vel -= gyro_bias;
    measurement.lin_accel -= accel_bias;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (!this->imuMeasFromTimeRange(
        imu_snapshot, start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) ||
      begin_imu_it == end_imu_it || begin_imu_it + 1 == end_imu_it) {
    result.status = ImuIntegrationStatus::kHistoryUnavailable;
    result.reason = "unable to bracket the LiDAR scan with IMU measurements";
    return result;
  }

  for (auto previous = begin_imu_it, current = begin_imu_it + 1;
       current != end_imu_it; ++previous, ++current) {
    const double dt = current->stamp - previous->stamp;
    if (!std::isfinite(previous->stamp) || !std::isfinite(current->stamp) ||
        !std::isfinite(dt) || dt <= 0.0 || dt > this->imu_max_gap_seconds_) {
      std::ostringstream reason;
      reason << "invalid IMU interval of " << dt << " seconds";
      result.status = ImuIntegrationStatus::kImuGap;
      result.reason = reason.str();
      return result;
    }
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.stamp - f1.stamp;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  result.anchor_velocity.setConstant(std::numeric_limits<float>::quiet_NaN());
  result.frames = this->integrateImuInternal(
    q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it,
    sorted_timestamps.size() / 2, result.anchor_velocity);
  if (result.frames.size() != sorted_timestamps.size() ||
      !result.anchor_velocity.allFinite() ||
      !std::all_of(result.frames.begin(), result.frames.end(), [](const Eigen::Matrix4f& frame) {
        return frame.allFinite();
      })) {
    result.frames.clear();
    result.status = ImuIntegrationStatus::kIncomplete;
    result.reason = "IMU integration did not produce a finite pose for every point timestamp";
    return result;
  }

  result.status = ImuIntegrationStatus::kSuccess;
  result.reason = "success";
  return result;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it,
                                     std::size_t anchor_index,
                                     Eigen::Vector3f& anchor_velocity) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();
  std::size_t timestamp_index = 0;

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    double dt = f.stamp - f0.stamp;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // Orientation
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation
      Eigen::Quaternionf q_i (
        q.w() - 0.5*( q.x()*omega_i[0] + q.y()*omega_i[1] + q.z()*omega_i[2] ) * idt,
        q.x() + 0.5*( q.w()*omega_i[0] - q.z()*omega_i[1] + q.y()*omega_i[2] ) * idt,
        q.y() + 0.5*( q.z()*omega_i[0] + q.w()*omega_i[1] - q.x()*omega_i[2] ) * idt,
        q.z() + 0.5*( q.x()*omega_i[1] - q.y()*omega_i[0] + q.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;
      Eigen::Vector3f v_i = v + a0*idt + 0.5*j*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      if (timestamp_index == anchor_index) {
        anchor_velocity = v_i;
      }

      stamp_it++;
      ++timestamp_index;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;

    prev_imu_it = imu_it;

  }

  return imu_se3;

}

void OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void OdomNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;
  const Eigen::Vector3f corrected_lin_accel =
    this->imu_meas.lin_accel - this->state.b.accel;
  const Eigen::Vector3f corrected_ang_vel =
    this->imu_meas.ang_vel - this->state.b.gyro;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(corrected_lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = corrected_ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = corrected_ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

}

void OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::msg::Imu::SharedPtr OdomNode::transformImu(
    const sensor_msgs::msg::Imu::SharedPtr& imu_raw, double dt) {

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();

  // Copy header
  imu->header = imu_raw->header;

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void OdomNode::computeSpaciousness() {

  // compute range of points
  std::vector<float> ds;

  for (int i = 0; i < this->original_scan->points.size(); i++) {
    float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
                        pow(this->original_scan->points[i].y, 2));
    ds.push_back(d);
  }

  // median
  std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
  float median_curr = ds[ds.size()/2];
  static float median_prev = median_curr;
  float median_lpf = 0.95*median_prev + 0.05*median_curr;
  median_prev = median_lpf;

  // push
  this->metrics.spaciousness.push_back( median_lpf );

}

void OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

  this->metrics.density.push_back( density_lpf );

}

void OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  auto cloud = std::make_shared<pcl::PointCloud<PointType>>();

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points = std::make_shared<pcl::PointCloud<PointType>>();
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = std::make_shared<pcl::PointIndices>();
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void OdomNode::updateKeyframes() {

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto& k : this->keyframes) {

    // calculate distance between current pose and pose in keyframes
    float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                          pow(this->state.p[1] - k.first.first[1], 2) +
                          pow(this->state.p[2] - k.first.first[2], 2) );

    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
      ++num_nearby;
    }

    // store into variable
    if (delta_d < closest_d) {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;

  }

  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                   pow(this->state.p[1] - closest_pose[1], 2) +
                   pow(this->state.p[2] - closest_pose[2], 2) );

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  if (newKeyframe) {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.push_back(this->scan_header_stamp);
    this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.push_back(this->T_corr);
    lock.unlock();

  }

}

void OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames) {

  // make sure dists is not empty
  if (!dists.size()) { return; }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn);

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex);

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }

  // get indices for top kNN for concave hull
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave);

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());

  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) {

      // create current submap cloud
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second;
      lock.unlock();

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // transform the new keyframe(s) and associated covariance list(s)
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    this->publishKeyframe(this->keyframes[i], this->keyframe_timestamps[i]);
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void OdomNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                                   + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                                     + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

} // namespace dlio

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(dlio::OdomNode)
