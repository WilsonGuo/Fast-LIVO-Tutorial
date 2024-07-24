
#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>

#ifdef USE_IKFOM
#include "use-ikfom.hpp"
#endif

/// *************Preconfiguration

#define MAX_INI_COUNT (200)

// 判断点的时间是否先后颠倒
const bool time_list(PointType &x, PointType &y);

/// *************IMU Process and undistortion

/**
 * IMU 处理和点云去畸变
 */
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void push_update_state(double offs_t, StatesGroup state);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4, 4) & T);
  void set_gyr_cov_scale(const V3D &scaler);
  void set_acc_cov_scale(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
#ifdef USE_IKFOM
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);
#else
  void Process(const LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
  void Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
  void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
#endif

  ros::NodeHandle nh;
  ofstream fout_imu;       // imu参数输出文件
  V3D cov_acc;             // 加速度测量协方差
  V3D cov_gyr;             // 角速度测量协方差
  V3D cov_acc_scale;       // 加速度缩放测量协方差
  V3D cov_gyr_scale;       // 角速度缩放测量协方差
  V3D cov_bias_gyr;        // 角速度测量协方差偏置
  V3D cov_bias_acc;        // 加速度测量协方差偏置
  double first_lidar_time; // 当前帧第一个点云时间

private:
#ifdef USE_IKFOM
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
#else
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void Forward(const MeasureGroup &meas, StatesGroup &state_inout, double pcl_beg_time, double end_time);
  void Backward(const LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
#endif

  PointCloudXYZI::Ptr cur_pcl_un_;        // 当前帧点云未去畸变
  sensor_msgs::ImuConstPtr last_imu_;     // 上一帧imu
  deque<sensor_msgs::ImuConstPtr> v_imu_; // imu队列
  vector<Pose6D> IMUpose;                 // imu位姿
  vector<M3D> v_rot_pcl_;                 // 未使用
  M3D Lid_rot_to_IMU;                     // lidar到IMU的旋转外参
  V3D Lid_offset_to_IMU;                  // lidar到IMU的位置外参
  V3D mean_acc;                           // 加速度均值,用于计算方差
  V3D mean_gyr;                           // 角速度均值，用于计算方差
  V3D angvel_last;                        // 上一帧角速度
  V3D acc_s_last;                         // 上一帧加速度
  V3D last_acc;
  V3D last_ang;
  double start_timestamp_;     // 开始时间戳
  double last_lidar_end_time_; // 上一帧结束时间戳
  int init_iter_num = 1;       // 初始化迭代次数
  bool b_first_frame_ = true;  // 是否是第一帧
  bool imu_need_init_ = true;  // 是否需要初始化imu
};
#endif
