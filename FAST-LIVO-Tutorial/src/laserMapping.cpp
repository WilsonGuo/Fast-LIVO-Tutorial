// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
// #include <common_lib.h>
#include <image_transport/image_transport.h>
#include "IMU_Processing.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vikit/camera_loader.h>
#include "lidar_selection.h"

#ifdef USE_ikdtree
#ifdef USE_ikdforest
#include <ikd-Forest/ikd_Forest.h>
#else
#include <ikd-Tree/ikd_Tree.h>
#endif
#else
#include <pcl/kdtree/kdtree_flann.h>
#endif

#define INIT_TIME (0.5)
#define MAXN (360000)
#define PUBFRAME_PERIOD (20)

float DET_RANGE = 300.0f; // 设置的当前雷达系中心到各个地图边缘的距离
#ifdef USE_ikdforest
const int laserCloudWidth = 200;
const int laserCloudHeight = 200;
const int laserCloudDepth = 200;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;
#else
const float MOV_THRESHOLD = 1.5f; // 设置的当前雷达系中心到各个地图边缘的权重
#endif

mutex mtx_buffer;              // 互斥锁
condition_variable sig_buffer; // 条件变量

string root_dir = ROOT_DIR;                                         // 设置根目录
string map_file_path, lid_topic, imu_topic, img_topic, config_file; // 设置地图文件路径，雷达topic，imu topic
M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

Vector3d Lidar_offset_to_IMU; // 雷达与IMU的外参

// 设置迭代次数，下采样的点数，最大迭代次数，有效点数
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0;

// 设置有效特征点数，时间log计数器，publish_count：接收到的IMU的Msg的总数
int effct_feat_num = 0, time_log_counter = 0, publish_count = 0;

int MIN_IMG_COUNT = 0;

// 设置残差平均值
double res_mean_last = 0.05;

// 设置imu的角速度协方差，加速度协方差
double gyr_cov_scale = 0, acc_cov_scale = 0;

// 设置雷达时间戳，imu时间戳
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;

// 设置滤波器的最小尺寸，地图的最小尺寸，视野角度
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;

// 设置立方体长度，视野一半的角度，视野总角度，总距离，雷达结束时间，雷达初始时间
double cube_len = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;

double first_img_time = -1.0;

// kdtree_incremental_time为kdtree建立时间，kdtree_search_time为kdtree搜索时间，kdtree_delete_time为kdtree删除时间;
double kdtree_incremental_time = 0, kdtree_search_time = 0, kdtree_delete_time = 0.0;

// kdtree_search_counter为ikd-tree搜索数量 kdtree_size_st为ikd-tree获得的节点数，kdtree_size_end为ikd-tree结束时的节点数，add_point_size为添加点的数量，kdtree_delete_counter为删除点的数量
int kdtree_search_counter = 0, kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
;
// double copy_time, readd_time, fov_check_time, readd_box_time, delete_box_time;
double copy_time = 0, readd_time = 0, fov_check_time = 0, readd_box_time = 0, delete_box_time = 0;

// T1为雷达初始时间戳，s_plot为整个流程耗时，s_plot2特征点数量,s_plot3为kdtree增量时间，s_plot4为kdtree搜索耗时，s_plot5为kdtree删除点数量，s_plot6为kdtree删除耗时，s_plot7为kdtree初始大小，s_plot8为kdtree结束大小,s_plot9为平均消耗时间，s_plot10为添加点数量，s_plot11为点云预处理的总时间
double T1[MAXN], T2[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];

// 定义全局变量，用于记录时间,match_time为匹配时间，solve_time为求解时间，solve_const_H_time为求解H矩阵时间
double match_time = 0, solve_time = 0, solve_const_H_time = 0;

bool lidar_pushed, flg_reset, flg_exit = false;
bool ncc_en;
int dense_map_en = 1;
int img_en = 1;
int lidar_en = 1;
int debug = 0;
bool fast_lio_is_ready = false;
int grid_size, patch_size;
double outlier_threshold, ncc_thre;
double delta_time = 0.0;

vector<BoxPointType> cub_needrm;              // ikd-tree中，地图需要移除的包围盒序列
vector<BoxPointType> cub_needad;              // ikd-tree中，地图需要添加的包围盒序列
deque<PointCloudXYZI::Ptr> lidar_buffer;      // 记录特征提取或间隔采样后的lidar（特征）数据
deque<double> time_buffer;                    // 激光雷达数据时间戳缓存队列
deque<sensor_msgs::Imu::ConstPtr> imu_buffer; // IMU数据缓存队列
deque<cv::Mat> img_buffer;
deque<double> img_time_buffer; // 图像数据时间戳缓存队列
vector<bool> point_selected_surf;
vector<vector<int>> pointSearchInd_surf; // 每个点的索引,暂时没用到
vector<PointVector> Nearest_Points;      // 每个点的最近点序列

vector<double> res_last;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> cameraextrinT(3, 0.0);
vector<double> cameraextrinR(9, 0.0);
double total_residual;
double LASER_POINT_COV, IMG_POINT_COV, cam_fx, cam_fy, cam_cx, cam_cy;
bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
// surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI()); // 提取地图中的特征点，IKD-tree获得
PointCloudXYZI::Ptr cube_points_add(new PointCloudXYZI());
PointCloudXYZI::Ptr map_cur_frame_point(new PointCloudXYZI());
PointCloudXYZI::Ptr sub_map_cur_frame_point(new PointCloudXYZI());

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());  // 去畸变的特征
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  // 畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); // 畸变纠正后降采样的单帧点云，w系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());          // 特征点在地图中对应点的，局部平面参数,w系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());    // laserCloudOri是畸变纠正后降采样的单帧点云，body系
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());    // 对应点法相量

pcl::VoxelGrid<PointType> downSizeFilterSurf; // 单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;  // 未使用

#ifdef USE_ikdtree
#ifdef USE_ikdforest
KD_FOREST ikdforest;
#else
KD_TREE ikdtree; // ikd-tree类
#endif
#else
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());
#endif

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);  // 雷达相对于body系的X轴方向的点
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0); // 雷达相对于world系的X轴方向的点
V3D euler_cur;                                // 当前的欧拉角
V3D position_last(Zero3d);                    // 上一帧的位置
Eigen::Matrix3d Rcl;                          // 旋转向量矩阵
Eigen::Vector3d Pcl;                          // 位移向量矩阵

// estimator inputs and output;
LidarMeasureGroup LidarMeasures;
// SparseMap sparse_map;
#ifdef USE_IKFOM
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 状态，噪声维度，输入
state_ikfom state_point;                         // 状态
vect3 pos_lid;                                   // world系下lidar坐标
#else
StatesGroup state;
#endif

// 输出的参数
nav_msgs::Path path;                      // 包含了一系列位姿
nav_msgs::Odometry odomAftMapped;         // 只包含了一个位姿
geometry_msgs::Quaternion geoQuat;        // 四元数
geometry_msgs::PoseStamped msg_body_pose; // 位姿

// 激光处理操作
shared_ptr<Preprocess> p_pre(new Preprocess()); // 定义指向激光雷达数据的预处理类Preprocess的智能指针

PointCloudXYZRGB::Ptr pcl_wait_save(new PointCloudXYZRGB());   // add save rbg map
PointCloudXYZI::Ptr pcl_wait_save_lidar(new PointCloudXYZI()); // add save xyzi map

bool pcd_save_en = true;
int pcd_save_interval = 20, pcd_index = 0;

// 按下ctrl+c后唤醒所有线程
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all(); // 会唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞
}

/**
 *
 *  将lio信息打印到log中
 */
inline void dump_lio_state_to_log(FILE *fp)
{
#ifdef USE_IKFOM
    // state_ikfom write_state = kf.get_x();
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
#else
    V3D rot_ang(Log(state.rot_end));
    fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1), state.pos_end(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1), state.vel_end(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1), state.bias_g(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state.bias_a(0), state.bias_a(1), state.bias_a(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state.gravity(0), state.gravity(1), state.gravity(2)); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
#endif
}

#ifdef USE_IKFOM
// 把点从body系转到world系，通过ikfom的位置和姿态
void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}
#endif
// 把点从body系转到world系
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
#endif

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}
// 把点从body系转到world系
template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
#endif
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}
// 含有RGB的点云从body系转到world系
void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
#endif
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity * 10000;
}

#ifndef USE_ikdforest
int points_cache_size = 0;

/**
 *
 * 得到被剔除的点
 */
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history); // 返回被剔除的点
    points_cache_size = points_history.size();      // 获取点的数量
}
#endif

BoxPointType get_cube_point(float center_x, float center_y, float center_z)
{
    BoxPointType cube_points;
    V3F center_p(center_x, center_y, center_z);
    // cout<<"center_p: "<<center_p.transpose()<<endl;

    for (int i = 0; i < 3; i++)
    {
        cube_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
        cube_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
    }

    return cube_points;
}

BoxPointType get_cube_point(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)
{
    BoxPointType cube_points;
    cube_points.vertex_max[0] = xmax;
    cube_points.vertex_max[1] = ymax;
    cube_points.vertex_max[2] = zmax;
    cube_points.vertex_min[0] = xmin;
    cube_points.vertex_min[1] = ymin;
    cube_points.vertex_min[2] = zmin;
    return cube_points;
}

#ifndef USE_ikdforest

// 在拿到eskf前馈结果后。动态调整地图区域，防止地图过大而内存溢出，类似LOAM中提取局部地图的方法
BoxPointType LocalMap_Points;      // ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false; // 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // X轴分界点转换到w系下，好像没有用到
#ifdef USE_IKFOM
    // state_ikfom fov_state = kf.get_x();
    // V3D pos_LiD = fov_state.pos + fov_state.rot * fov_state.offset_T_L_I;
    V3D pos_LiD = pos_lid; // global系下lidar位置
#else
    V3D pos_LiD = state.pos_end; // global系下lidar位置
#endif
    if (!Localmap_Initialized) // 初始化局部地图包围盒角点，以为w系下lidar位置为中心,得到长宽高200*200*200的局部地图
    {
        // 系统起始需要初始化局部地图的大小和位置
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;

    // 当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);

        // 与某个方向上的边界距离（例如1.5*300m）太小，标记需要移除need_move，参考论文Fig3
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }

    // 不需要挪动就直接退回了
    if (!need_move)
        return;

    // 否则需要计算移动的距离
    BoxPointType New_LocalMap_Points, tmp_boxpoints;

    // 新的局部地图盒子边界点
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;

        // 与包围盒最小值边界点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();

    // 使用Boxs删除指定盒内的点
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}
#endif

/*******使用topic从各个传感器中读取数据并传入预处理的过程。同时timediff_lidar_wrt_imu会完成将imu的时间戳对齐到激光雷达时间戳的功能，通过两个最新的时间戳完成校验以及对齐****************/

/**
 * 除了AVIA类型之外的雷达点云回调函数，将数据引入到buffer当中
 */
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock(); // 加锁
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);                         // 点云预处理
    lidar_buffer.push_back(ptr);                      // 将点云放入缓冲区
    time_buffer.push_back(msg->header.stamp.toSec()); // 将时间放入缓冲区
    last_timestamp_lidar = msg->header.stamp.toSec(); // 记录最后一个时间
    mtx_buffer.unlock();
    sig_buffer.notify_all(); // 唤醒所有线程
}

/**
 * 订阅器sub_pcl的回调函数：接收Livox激光雷达的点云数据，对点云数据进行预处理（特征提取、降采样、滤波），并将处理后的数据保存到激光雷达数据队列中
 */
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    // 互斥锁
    mtx_buffer.lock();

    // 如果当前扫描的激光雷达数据的时间戳比上一次扫描的激光雷达数据的时间戳早，需要将激光雷达数据缓存队列清空
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);                         // 对激光雷达数据进行预处理（特征提取或者降采样），其中p_pre是Preprocess类的智能指针
    lidar_buffer.push_back(ptr);                      // 将点云放入缓冲区
    time_buffer.push_back(msg->header.stamp.toSec()); // 将时间放入缓冲区
    last_timestamp_lidar = msg->header.stamp.toSec(); // 记录最后一个时间

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/**
 * 订阅器sub_imu的回调函数：接收IMU数据，将IMU数据保存到IMU数据缓存队列中
 */
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();
    // 如果当前IMU的时间戳小于上一个时刻IMU的时间戳，则IMU数据有误，将IMU数据缓存队列清空
    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp; // 将当前的IMU数据保存到IMU数据缓存队列中

    imu_buffer.push_back(msg);
    mtx_buffer.unlock(); // 解锁
    sig_buffer.notify_all();
}

/**
 * 获取图像数据
 */
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat img;
    img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
    return img;
}

void img_cbk(const sensor_msgs::ImageConstPtr &msg)
{
    if (!img_en) // 如果不使用相机，直接返回
    {
        return;
    }
    double msg_header_time = msg->header.stamp.toSec() + delta_time; // 相机采样时间+纠偏时长(+100或-100)
    if (msg_header_time < last_timestamp_img)
    {
        ROS_ERROR("img loop back, clear buffer");
        img_buffer.clear();
        img_time_buffer.clear();
    }
    mtx_buffer.lock();

    img_buffer.push_back(getImageFromMsg(msg));
    img_time_buffer.push_back(msg_header_time);
    last_timestamp_img = msg_header_time;

    mtx_buffer.unlock(); // 解锁
    sig_buffer.notify_all();
}
/*******主要处理了buffer中的数据，将两帧激光雷达点云数据时间内的IMU数据从缓存队列中取出，进行时间对齐，并保存到meas中*********/

/**
 * 同步数据
 */
bool sync_packages(LidarMeasureGroup &meas)
{
    if ((lidar_buffer.empty() && img_buffer.empty())) // 如果雷达缓存队列和相机缓存队列里面同时都是空的，就直接返回
    {
        return false;
    }

    if (meas.is_lidar_end) // If meas.is_lidar_end==true, means it just after scan end, clear all buffer in meas.
    {
        meas.measures.clear();
        meas.is_lidar_end = false;
    }

    // 如果还没有把雷达数据放到meas中的话，就执行一下操作
    if (!lidar_pushed)
    {
        if (lidar_buffer.empty()) // 如果雷达缓存队列为空，直接退出
        {
            return false;
        }
        meas.lidar = lidar_buffer.front();  // 取出缓存队列的第一帧雷达数据
        if (meas.lidar->points.size() <= 1) // 如果该lidar没有点云，则先清理队列，再返回false
        {
            mtx_buffer.lock();
            if (img_buffer.size() > 0) // temp method, ignore img topic when no lidar points, keep sync
            {
                lidar_buffer.pop_front();
                img_buffer.pop_front();
            }
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            return false;
        }
        sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list);                     // 通过采样时间戳来排序
        meas.lidar_beg_time = time_buffer.front();                                                 // 获取开始时间
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // 计算雷达扫描结束时间
        lidar_pushed = true;                                                                       // flag
    }

    if (img_buffer.empty()) // 如果相机缓存队列为空，说明没有使用相机或一直没接收到相机数据
    {
        if (last_timestamp_imu < lidar_end_time + 0.02) // imu时间必须大于雷达采样时间
        {                                               // imu message needs to be larger than lidar_end_time, keep complete propagate.
            return false;
        }
        struct MeasureGroup m; // standard method to keep imu message.
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        mtx_buffer.lock();

        // 拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
        while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) // 如果imu缓存队列中的数据时间戳小于雷达结束时间戳，则将该数据放到meas中,代表了这一帧中的imu数据
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time)
                break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lidar_buffer.pop_front(); // 将lidar数据弹出
        time_buffer.pop_front();  // 将时间戳弹出
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        lidar_pushed = false;     // 将lidar_pushed置为false，代表lidar数据已经被放到meas中了
        meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
        meas.measures.push_back(m);
        return true; // 退出该函数
    }

    // 如果有相机或接收到了相机数据
    struct MeasureGroup m;

    if ((img_time_buffer.front() > lidar_end_time)) // 图像的时间戳大于雷达扫描结束时间，先要再一次处理雷达数据
    {                                               // has img topic, but img topic timestamp larger than lidar end time, process lidar topic.
        if (last_timestamp_imu < lidar_end_time + 0.02)
        {
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec(); // 取出imu缓存队列的第一帧数据的时间戳
        m.imu.clear();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) // 如果imu缓存队列中的数据时间戳小于雷达结束时间戳，则将该数据放到meas中,代表了这一帧中的imu数据
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time)
                break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        lidar_pushed = false;
        meas.is_lidar_end = true;
        meas.measures.push_back(m);
    }
    else
    {
        double img_start_time = img_time_buffer.front(); // 获取图像缓存队列的第一帧数据的时间戳
        if (last_timestamp_imu < img_start_time)
        {
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        m.imu.clear();
        m.img_offset_time = img_start_time - meas.lidar_beg_time; // 获取图像的偏移时长 record img offset time, it shoule be the Kalman update timestamp.
        m.img = img_buffer.front();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time < img_start_time))) // 如果imu缓存队列中的数据时间戳小于图像结束时间戳，则将该数据放到meas中,代表了这一帧中的imu数据
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > img_start_time)
                break;
            m.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        meas.is_lidar_end = false; // has img topic in lidar scan, so flag "is_lidar_end=false"
        meas.measures.push_back(m);
    }
    return true;
}

/**
 * 生成增量地图
 */
void map_incremental()
{
    for (int i = 0; i < feats_down_size; i++)
    {
        /* 转换到世界坐标系 */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    }
#ifdef USE_ikdtree
#ifdef USE_ikdforest
    ikdforest.Add_Points(feats_down_world->points, lidar_end_time);
#else
    ikdtree.Add_Points(feats_down_world->points, true);
#endif
#endif
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());

/**
 * 发布彩色点云（含RGB信息）
 */
void publish_frame_world_rgb(const ros::Publisher &pubLaserCloudFullRes, lidar_selection::LidarSelectorPtr lidar_selector)
{

    uint size = pcl_wait_pub->points.size();
    PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
    if (img_en)
    {
        laserCloudWorldRGB->clear();
        cv::Mat img_rgb = lidar_selector->img_rgb;
        for (int i = 0; i < size; i++)
        {
            PointTypeRGB pointRGB;
            pointRGB.x = pcl_wait_pub->points[i].x;
            pointRGB.y = pcl_wait_pub->points[i].y;
            pointRGB.z = pcl_wait_pub->points[i].z;
            V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
            V2D pc(lidar_selector->new_frame_->w2c(p_w));
            if (lidar_selector->new_frame_->cam_->isInFrame(pc.cast<int>(), 0))
            {
                V3F pixel = lidar_selector->getpixel(img_rgb, pc);
                pointRGB.r = pixel[2];
                pointRGB.g = pixel[1];
                pointRGB.b = pixel[0];
                laserCloudWorldRGB->push_back(pointRGB);
            }
        }
    }

    if (1) // if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        if (img_en)
        {
            // cout<<"RGB pointcloud size: "<<laserCloudWorldRGB->size()<<endl;
            pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
        }
        else
        {
            pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        }
        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
    // mtx_buffer_pointcloud.unlock();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
        *pcl_wait_save += *laserCloudWorldRGB;
}

/**
 * 发布简单的点云数据（仅仅包含位置信息）
 */
void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes)
{
    uint size = pcl_wait_pub->points.size();
    if (1)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;

        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
    if (pcd_save_en)
        *pcl_wait_save_lidar += *pcl_wait_pub;
}

/**
 * 发布可视化的点云数据（带有强度值）
 */
void publish_visual_world_map(const ros::Publisher &pubVisualCloud)
{
    PointCloudXYZI::Ptr laserCloudFullRes(map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size == 0)
        return;
    PointCloudXYZI::Ptr pcl_visual_wait_pub(new PointCloudXYZI());
    *pcl_visual_wait_pub = *laserCloudFullRes;
    if (1)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}
/**
 * 发布可视化的点云数据（带有强度值）的增量子图
 */
void publish_visual_world_sub_map(const ros::Publisher &pubSubVisualCloud)
{
    PointCloudXYZI::Ptr laserCloudFullRes(sub_map_cur_frame_point);
    int size = laserCloudFullRes->points.size();
    if (size == 0)
        return;
    PointCloudXYZI::Ptr sub_pcl_visual_wait_pub(new PointCloudXYZI());
    *sub_pcl_visual_wait_pub = *laserCloudFullRes;
    if (1)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*sub_pcl_visual_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
        pubSubVisualCloud.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

/**
 * 发布世界坐标系中生效的特征点
 */
void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

/**
 * 发布ikd-tree地图
 */
void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time::now();
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

// 设置输出的t,q，在publish_odometry，publish_path调用
template <typename T>
void set_posestamp(T &out)
{
#ifdef USE_IKFOM
    // state_ikfom stamp_state = kf.get_x();
    out.position.x = state_point.pos(0); // 将eskf求得的位置传入
    out.position.y = state_point.pos(1);
    out.position.z = state_point.pos(2);
#else
    out.position.x = state.pos_end(0); // 将求得的位置传入
    out.position.y = state.pos_end(1);
    out.position.z = state.pos_end(2);
#endif
    out.orientation.x = geoQuat.x; // 将求得的姿态传入
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

/**
 * 发布里程计信息
 */
void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
    set_posestamp(odomAftMapped.pose.pose);
    pubOdomAftMapped.publish(odomAftMapped);
}

void publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_odom_frame";
    set_posestamp(msg_body_pose.pose);
    mavros_pose_publisher.publish(msg_body_pose);
}

/**
 * 每隔10个发布一下位姿
 */
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
}

#ifdef USE_IKFOM
// 计算残差信息
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
/** 最接近曲面搜索和残差计算 **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    // 对降采样后的每个特征点进行残差计算
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];   // 获取降采样后的每个特征点
        PointType &point_world = feats_down_world->points[i]; // 获取降采样后的每个特征点的世界坐标
        // double search_start = omp_get_wtime();
        /* transform to world frame */
        // 将点转换至世界坐标系下
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // 将点转换至世界坐标系下,从而来计算残差
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
#ifdef USE_ikdtree
        auto &points_near = Nearest_Points[i];
#else
        auto &points_near = pointSearchInd_surf[i];
#endif

        if (ekfom_data.converge) // 如果收敛了
        {
            // 在已构造的地图上查找特征点的最近邻
#ifdef USE_ikdtree
#ifdef USE_ikdforest

            uint8_t search_flag = 0;
            search_flag = ikdforest.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, first_lidar_time, 5);
#else
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
#endif
#else
            kdtreeSurfFromMap->nearestKSearch(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
#endif
            // 如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，则认为该点不是有效点
            point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;

#ifdef USE_ikdforest
            point_selected_surf[i] = point_selected_surf[i] && (search_flag == 0);
#endif
        }

        // kdtree_search_time += omp_get_wtime() - search_start;

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;                          // 如果该点不是有效点
        point_selected_surf[i] = false; // 将该点设置为无效点，用来计算是否为平面点

        // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
        if (esti_plane(pabcd, points_near, 0.1f)) // 找平面点法向量寻找
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); // 计算点到平面的距离
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());                                                   // 计算残差

            if (s > 0.9) // 如果残差大于阈值，则认为该点是有效点
            {
                point_selected_surf[i] = true;   // 再次回复为有效点
                normvec->points[i].x = pabcd(0); // 将法向量存储至normvec
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; // 将点到平面的距离存储至normvec的intensit中
                res_last[i] = abs(pd2);             // 将残差存储至res_last
            }
        }
    }

    effct_feat_num = 0; // 有效特征点数

    for (int i = 0; i < feats_down_size; i++)
    {
        // 根据point_selected_surf状态判断哪些点是可用的
        if (point_selected_surf[i] && (res_last[i] <= 2.0))
        {
            // body点存到laserCloudOri中
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // 将降采样后的每个特征点存储至laserCloudOri
            corr_normvect->points[effct_feat_num] = normvec->points[i];         // 拟合平面点存到corr_normvect中
            total_residual += res_last[i];                                      // 计算总残差
            effct_feat_num++;                                                   // 有效特征点数加1
        }
    }

    res_mean_last = total_residual / effct_feat_num; // 计算残差平均值
    match_time += omp_get_wtime() - match_start;     // 返回从匹配开始时候所经过的时间
    double solve_start_ = omp_get_wtime();           // 下面是solve求解的时间

    // 测量雅可比矩阵H和测量向量的计算 H=J*P*J'
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 测量雅可比矩阵H
    ekfom_data.h.resize(effct_feat_num);                 // 测量向量h

    // 求观测值与误差的雅克比矩阵
    for (int i = 0; i < effct_feat_num; i++)
    {
        // 拿到的有效点的坐标
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;                                            // 计算点的叉矩阵
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);               // 从点值转换到叉乘矩阵
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // 转换到IMU坐标系下, offset_R_L_I，offset_T_L_I为IMU的旋转姿态和位移,此时转到了IMU坐标系下
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this); // 计算imu中点的叉矩阵

        // 得到对应的曲面/角的法向量
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        // 计算测量雅可比矩阵H
        V3D C(s.rot.conjugate() * norm_vec);                       // 旋转矩阵的转置与法向量相乘得到C
        V3D A(point_crossmat * C);                                 // 对imu的差距真与C相乘得到A
        V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // 对点的差距真与C相乘得到B
        ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);

        // 测量:到最近表面/角落的距离
        ekfom_data.h(i) = -norm_p.intensity; // 点到面的距离
    }
    solve_time += omp_get_wtime() - solve_start_; // 返回从solve开始时候所经过的时间
}
#endif

/**
 *  读取外部设置的参数
 */
void readParameters(ros::NodeHandle &nh)
{
    nh.param<int>("dense_map_enable", dense_map_en, 1);
    nh.param<int>("img_enable", img_en, 1);
    nh.param<int>("lidar_enable", lidar_en, 1);
    nh.param<int>("debug", debug, 0);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4); // 卡尔曼滤波的最大迭代次数
    nh.param<bool>("ncc_en", ncc_en, false);
    nh.param<int>("min_img_count", MIN_IMG_COUNT, 1000);
    nh.param<double>("laserMapping/cam_fx", cam_fx, 400);
    nh.param<double>("laserMapping/cam_fy", cam_fy, 400);
    nh.param<double>("laserMapping/cam_cx", cam_cx, 300);
    nh.param<double>("laserMapping/cam_cy", cam_cy, 300);
    nh.param<double>("laser_point_cov", LASER_POINT_COV, 0.001);
    nh.param<double>("img_point_cov", IMG_POINT_COV, 10);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<string>("camera/img_topic", img_topic, "/left_camera/image");
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5); // 降采样时的体素大小
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);     // 降采样时的体素大小
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);       // 降采样时的体素大小
    nh.param<double>("cube_side_length", cube_len, 200);                 // 地图的局部区域的长度
    nh.param<double>("mapping/gyr_cov_scale", gyr_cov_scale, 1.0);
    nh.param<double>("mapping/acc_cov_scale", acc_cov_scale, 1.0);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);                   // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);            // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);                  // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);              // 采样间隔，即每隔point_filter_num个点取1个点
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, 0);        // 是否提取特征点（默认不进行特征点提取）
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 雷达相对于IMU的外参R
    nh.param<vector<double>>("camera/Pcl", cameraextrinT, vector<double>());    // 相机的平移外参
    nh.param<vector<double>>("camera/Rcl", cameraextrinR, vector<double>());    // 相机的旋转外参
    nh.param<int>("grid_size", grid_size, 40);
    nh.param<int>("patch_size", patch_size, 4);
    nh.param<double>("outlier_threshold", outlier_threshold, 100);
    nh.param<double>("ncc_thre", ncc_thre, 100);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<double>("delta_time", delta_time, 0.0);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    readParameters(nh);
    pcl_wait_pub->clear();

    // 订阅点云数据
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);

    // 订阅IMU数据
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    // 订阅图像数据
    ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);

    image_transport::Publisher img_pub = it.advertise("/rgb_img", 1);

    // 彩色点云发布器
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
    // 强度点云发布器
    ros::Publisher pubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_map", 100);
    // 点云子图发布器
    ros::Publisher pubSubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map", 100);
    // 特征点发布器
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
    // 点云地图发布器
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    // 里程计发布器
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);

    // 路径发布器
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 10);

#ifdef DEPLOY
    ros::Publisher mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
#endif

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

/*** variables definition ***/
#ifndef USE_IKFOM
    VD(DIM_STATE)
    solution;
    MD(DIM_STATE, DIM_STATE)
    G, H_T_H, I_STATE;
    V3D rot_add, t_add;
    StatesGroup state_propagat;
    PointType pointOri, pointSel, coeff;
#endif
    int effect_feat_num = 0, frame_num = 0; // effect_feat_num:后面的代码中没有用到该变量  ,frame_num 雷达总帧数

    // aver_time_consu 每帧平均的处理总时间 * aver_time_icp 每帧中icp的平均时间 * aver_time_match 每帧中匹配的平均时间 * aver_time_incre 每帧中ikd-tree增量处理的平均时间 * aver_time_solve 每帧中计算的平均时间 * aver_time_const_H_time 每帧中计算的平均时间（当H恒定时） ,其他参数均未用到
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

    // 设置降采样参数
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
#ifdef USE_ikdforest
    ikdforest.Set_balance_criterion_param(0.6);
    ikdforest.Set_delete_criterion_param(0.5);
    ikdforest.Set_environment(laserCloudDepth, laserCloudWidth, laserCloudHeight, cube_len);
    ikdforest.Set_downsample_param(filter_size_map_min);
#endif

    shared_ptr<ImuProcess> p_imu(new ImuProcess()); // 其中p_imu为ImuProcess的智能指针,即，在此初始化了一个ImuProcess的对象实例
    V3D extT;
    M3D extR;
    extT << VEC_FROM_ARRAY(extrinT);
    extR << MAT_FROM_ARRAY(extrinR);
    Lidar_offset_to_IMU = extT; // 雷达与IMU的外参赋值

    lidar_selection::LidarSelectorPtr lidar_selector(new lidar_selection::LidarSelector(grid_size, new SparseMap));
    if (!vk::camera_loader::loadFromRosNs("laserMapping", lidar_selector->cam))
        throw std::runtime_error("Camera model not correctly specified.");
    lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
    lidar_selector->debug = debug;
    lidar_selector->patch_size = patch_size;
    lidar_selector->outlier_threshold = outlier_threshold;
    lidar_selector->ncc_thre = ncc_thre;
    lidar_selector->sparse_map->set_camera2lidar(cameraextrinR, cameraextrinT);
    lidar_selector->set_extrinsic(extT, extR);
    lidar_selector->state = &state;
    lidar_selector->state_propagat = &state_propagat;
    lidar_selector->NUM_MAX_ITERATIONS = NUM_MAX_ITERATIONS;
    lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
    lidar_selector->img_point_cov = IMG_POINT_COV;
    lidar_selector->fx = cam_fx;
    lidar_selector->fy = cam_fy;
    lidar_selector->cx = cam_cx;
    lidar_selector->cy = cam_cy;
    lidar_selector->ncc_en = ncc_en;
    lidar_selector->init();

    p_imu->set_extrinsic(extT, extR);
    p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
    p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
    p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));

#ifndef USE_IKFOM
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
#endif

#ifdef USE_IKFOM
    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
#endif
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);

#ifdef USE_ikdforest
    ikdforest.Set_balance_criterion_param(0.6);
    ikdforest.Set_delete_criterion_param(0.5);
#endif
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle); // 中断处理函数，如果有中断信号（比如Ctrl+C），则执行第二个参数里面的SigHandle函数

    ros::Rate rate(5000); // 设置ROS程序主循环每次运行的时间至少为0.0002秒（5000Hz）
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();

        // 将激光雷达点云数据和IMU数据从缓存队列中取出，进行时间对齐，并保存到Measures中
        if (!sync_packages(LidarMeasures))
        {
            status = ros::ok();
            cv::waitKey(1);
            rate.sleep();
            continue;
        }

        // 激光雷达第一次扫描
        if (flg_reset)
        {
            ROS_WARN("reset when rosbag play back");
            p_imu->Reset();
            flg_reset = false;
            continue;
        }

        double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

        match_time = kdtree_search_time = kdtree_search_counter = solve_time = solve_const_H_time = svd_time = 0;
        t0 = omp_get_wtime();
#ifdef USE_IKFOM
        // 对IMU数据进行预处理，其中包含了点云畸变处理 前向传播 反向传播
        p_imu->Process(LidarMeasures, kf, feats_undistort);

        // 获取kf预测的全局状态（imu）
        state_point = kf.get_x();

        // 世界系下雷达坐标系的位置
        // 下面式子的意义是W^p_L = W^p_I + W^R_I * I^t_L
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
#else
        p_imu->Process2(LidarMeasures, state, feats_undistort);
        state_propagat = state;
#endif

        if (lidar_selector->debug)
        {
            LidarMeasures.debug_show();
        }

        if (feats_undistort->empty() || (feats_undistort == nullptr)) // 如果点云数据为空，则代表了激光雷达没有完成去畸变，此时还不能初始化成功
        {
            // cout<<" No point!!!"<<endl;
            if (!fast_lio_is_ready)
            {
                first_lidar_time = LidarMeasures.lidar_beg_time; // 记录第一次扫描的时间
                p_imu->first_lidar_time = first_lidar_time;      // 将第一帧的时间传给imu作为当前帧的第一个点云时间
                LidarMeasures.measures.clear();
                cout << "FAST-LIO not ready" << endl;
                continue;
            }
        }
        else
        {
            int size = feats_undistort->points.size();
        }
        fast_lio_is_ready = true;
        flg_EKF_inited = (LidarMeasures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true; // 判断是否初始化完成，需要满足第一次扫描的时间和第一个点云时间的差值大于INIT_TIME

        if (!LidarMeasures.is_lidar_end)
        {
            cout << "[ VIO ]: Raw feature num: " << pcl_wait_pub->points.size() << "." << endl;
            if (first_lidar_time < 10)
            {
                continue;
            }

            if (img_en)
            {
                euler_cur = RotMtoEuler(state.rot_end);
                fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose()
                         << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << endl;

                lidar_selector->detect(LidarMeasures.measures.back().img, pcl_wait_pub);
                // int size = lidar_selector->map_cur_frame_.size();
                int size_sub = lidar_selector->sub_map_cur_frame_.size();

                // map_cur_frame_point->clear();
                sub_map_cur_frame_point->clear();

                for (int i = 0; i < size_sub; i++)
                {
                    PointType temp_map;
                    temp_map.x = lidar_selector->sub_map_cur_frame_[i]->pos_[0];
                    temp_map.y = lidar_selector->sub_map_cur_frame_[i]->pos_[1];
                    temp_map.z = lidar_selector->sub_map_cur_frame_[i]->pos_[2];
                    temp_map.intensity = 0.;
                    sub_map_cur_frame_point->push_back(temp_map);
                }
                cv::Mat img_rgb = lidar_selector->img_cp;
                cv_bridge::CvImage out_msg;
                out_msg.header.stamp = ros::Time::now();
                // out_msg.header.frame_id = "camera_init";
                out_msg.encoding = sensor_msgs::image_encodings::BGR8;
                out_msg.image = img_rgb;
                img_pub.publish(out_msg.toImageMsg());

                if (img_en)
                    publish_frame_world_rgb(pubLaserCloudFullRes, lidar_selector);
                publish_visual_world_sub_map(pubSubVisualCloud);

                geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
                publish_odometry(pubOdomAftMapped);
                euler_cur = RotMtoEuler(state.rot_end);
                fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose()
                         << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << " " << feats_undistort->points.size() << endl;
            }
            continue;
        }

/*** Segment the map in lidar FOV ***/
#ifndef USE_ikdforest
        lasermap_fov_segment(); // 动态调整局部地图,在拿到eskf前馈结果后
#endif
        /*** 在一个扫描周期内对特征点云进行降采样 ***/
        downSizeFilterSurf.setInputCloud(feats_undistort); // 获得去畸变后的点云数据
        downSizeFilterSurf.filter(*feats_down_body);       // 滤波降采样后的点云数据
#ifdef USE_ikdtree
/*** initialize the map kdtree ***/
#ifdef USE_ikdforest
        if (!ikdforest.initialized)
        {
            if (feats_down_body->points.size() > 5)
            {
                ikdforest.Build(feats_down_body->points, true, lidar_end_time);
            }
            continue;
        }
        int featsFromMapNum = ikdforest.total_size;
#else
        if (ikdtree.Root_Node == nullptr)
        {
            if (feats_down_body->points.size() > 5)
            {
                ikdtree.set_downsample_param(filter_size_map_min); // 设置ikd tree的降采样参数
                ikdtree.Build(feats_down_body->points);            // 将下采样得到的地图点大小于body系大小一致
            }
            continue;
        }
        int featsFromMapNum = ikdtree.size();
#endif
#else
        if (featsFromMap->points.empty())
        {
            downSizeFilterMap.setInputCloud(feats_down_body);
        }
        else
        {
            downSizeFilterMap.setInputCloud(featsFromMap);
        }
        downSizeFilterMap.filter(*featsFromMap);
        int featsFromMapNum = featsFromMap->points.size();
#endif
        feats_down_size = feats_down_body->points.size(); // 记录滤波后的点云数量

        /*** ICP and iterated Kalman filter update ***/
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        // vector<double> res_last(feats_down_size, 1000.0); // initial //
        res_last.resize(feats_down_size, 1000.0);

        t1 = omp_get_wtime();
        if (lidar_en)
        {
            euler_cur = RotMtoEuler(state.rot_end);
#ifdef USE_IKFOM
            // state_ikfom fout_state = kf.get_x();
            fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose() * 57.3 << " " << state_point.pos.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;
#else
            fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose()
                     << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << endl;
#endif
        }

#ifdef USE_ikdtree
        if (0)
        {
            PointVector().swap(ikdtree.PCL_Storage);                             // 释放PCL_Storage的内存
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD); // 把树展平用于展示
            featsFromMap->clear();
            featsFromMap->points = ikdtree.PCL_Storage;
        }
#else
        kdtreeSurfFromMap->setInputCloud(featsFromMap);
#endif

        point_selected_surf.resize(feats_down_size, true); // 搜索索引
        pointSearchInd_surf.resize(feats_down_size);       // 将降采样处理后的点云用于搜索最近点
        Nearest_Points.resize(feats_down_size);
        int rematch_num = 0;
        bool nearest_search_en = true; //

        t2 = omp_get_wtime();

/*** 迭代状态估计 ***/
#ifdef MP_EN
        printf("[ LIO ]: Using multi-processor, used core number: %d.\n", MP_PROC_NUM);
#endif
        double t_update_start = omp_get_wtime();
#ifdef USE_IKFOM
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); // 迭代卡尔曼滤波更新，更新地图信息
        state_point = kf.get_x();
        euler_cur = SO3ToEuler(state_point.rot); // 外参，旋转矩阵转欧拉角
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];
#else

        if (img_en)
        {
            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
            for (int i = 0; i < 1; i++)
            {
            }
        }

        if (lidar_en)
        {
            for (iterCount = -1; iterCount < NUM_MAX_ITERATIONS && flg_EKF_inited; iterCount++)
            {
                match_start = omp_get_wtime();
                PointCloudXYZI().swap(*laserCloudOri);
                PointCloudXYZI().swap(*corr_normvect);
                // laserCloudOri->clear();
                // corr_normvect->clear();
                total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
                omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
                // normvec->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    PointType &point_body = feats_down_body->points[i];
                    PointType &point_world = feats_down_world->points[i];
                    V3D p_body(point_body.x, point_body.y, point_body.z);
                    /* transform to world frame */
                    pointBodyToWorld(&point_body, &point_world); // 将下采样得到的地图点转换为世界坐标系下的点云
                    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
#ifdef USE_ikdtree
                    auto &points_near = Nearest_Points[i];
#else
                    auto &points_near = pointSearchInd_surf[i];
#endif
                    uint8_t search_flag = 0;
                    double search_start = omp_get_wtime();
                    if (nearest_search_en)
                    {
/** Find the closest surfaces in the map **/
#ifdef USE_ikdtree
#ifdef USE_ikdforest
                        search_flag = ikdforest.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis, first_lidar_time, 5);
#else
                        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
#endif
#else
                        kdtreeSurfFromMap->nearestKSearch(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
#endif

                        point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;

#ifdef USE_ikdforest
                        point_selected_surf[i] = point_selected_surf[i] && (search_flag == 0);
#endif
                        kdtree_search_time += omp_get_wtime() - search_start;
                        kdtree_search_counter++;
                    }

                    if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS)
                        continue;

                    VF(4)
                    pabcd;
                    point_selected_surf[i] = false;
                    if (esti_plane(pabcd, points_near, 0.1f)) //(planeValid)
                    {
                        float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                        float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                        if (s > 0.9)
                        {
                            point_selected_surf[i] = true;
                            normvec->points[i].x = pabcd(0);
                            normvec->points[i].y = pabcd(1);
                            normvec->points[i].z = pabcd(2);
                            normvec->points[i].intensity = pd2;
                            res_last[i] = abs(pd2);
                        }
                    }
                }
                // cout<<"pca time test: "<<pca_time1<<" "<<pca_time2<<endl;
                effct_feat_num = 0;
                laserCloudOri->resize(feats_down_size);
                corr_normvect->reserve(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    if (point_selected_surf[i] && (res_last[i] <= 2.0))
                    {
                        laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                        corr_normvect->points[effct_feat_num] = normvec->points[i];
                        total_residual += res_last[i];
                        effct_feat_num++;
                    }
                }

                res_mean_last = total_residual / effct_feat_num;
                match_time += omp_get_wtime() - match_start;
                solve_start = omp_get_wtime();

                /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                MatrixXd Hsub(effct_feat_num, 6);
                VectorXd meas_vec(effct_feat_num);

                for (int i = 0; i < effct_feat_num; i++)
                {
                    const PointType &laser_p = laserCloudOri->points[i];
                    V3D point_this(laser_p.x, laser_p.y, laser_p.z);
                    point_this += Lidar_offset_to_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);

                    /*** get the normal vector of closest surface/corner ***/
                    const PointType &norm_p = corr_normvect->points[i];
                    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                    /*** calculate the Measuremnt Jacobian matrix H ***/
                    V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
                    Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;

                    /*** Measuremnt: distance to the closest surface/corner ***/
                    meas_vec(i) = -norm_p.intensity;
                }
                solve_const_H_time += omp_get_wtime() - solve_start;

                MatrixXd K(DIM_STATE, effct_feat_num);

                EKF_stop_flg = false;
                flg_EKF_converged = false;

                /*** Iterative Kalman Filter Update ***/
                if (!flg_EKF_inited)
                {
                    cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
                    /*** only run in initialization period ***/
                    MatrixXd H_init(MD(9, DIM_STATE)::Zero());
                    MatrixXd z_init(VD(9)::Zero());
                    H_init.block<3, 3>(0, 0) = M3D::Identity();
                    H_init.block<3, 3>(3, 3) = M3D::Identity();
                    H_init.block<3, 3>(6, 15) = M3D::Identity();
                    z_init.block<3, 1>(0, 0) = -Log(state.rot_end);
                    z_init.block<3, 1>(0, 0) = -state.pos_end;

                    auto H_init_T = H_init.transpose();
                    auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T + 0.0001 * MD(9, 9)::Identity()).inverse();
                    solution = K_init * z_init;

                    // solution.block<9,1>(0,0).setZero();
                    // state += solution;
                    // state.cov = (MatrixXd::Identity(DIM_STATE, DIM_STATE) - K_init * H_init) * state.cov;

                    state.resetpose();
                    EKF_stop_flg = true;
                }
                else
                {
                    auto &&Hsub_T = Hsub.transpose();
                    auto &&HTz = Hsub_T * meas_vec;
                    H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;
                    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
                    MD(DIM_STATE, DIM_STATE) &&K_1 =
                        (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
                    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
                    auto vec = state_propagat - state;
                    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

                    int minRow, minCol;
                    if (0) // if(V.minCoeff(&minRow, &minCol) < 1.0f)
                    {
                        VD(6)
                        V = H_T_H.block<6, 6>(0, 0).eigenvalues().real();
                        cout << "!!!!!! Degeneration Happend, eigen values: " << V.transpose() << endl;
                        EKF_stop_flg = true;
                        solution.block<6, 1>(9, 0).setZero();
                    }

                    state += solution;

                    rot_add = solution.block<3, 1>(0, 0);
                    t_add = solution.block<3, 1>(3, 0);

                    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015))
                    {
                        flg_EKF_converged = true;
                    }

                    deltaR = rot_add.norm() * 57.3;
                    deltaT = t_add.norm() * 100;
                }
                euler_cur = RotMtoEuler(state.rot_end);

                /*** Rematch Judgement ***/
                nearest_search_en = false;
                if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
                {
                    nearest_search_en = true;
                    rematch_num++;
                }

                /*** Convergence Judgements and Covariance Update ***/
                if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1)))
                {
                    if (flg_EKF_inited)
                    {
                        /*** Covariance Update ***/

                        state.cov = (I_STATE - G) * state.cov;
                        total_distance += (state.pos_end - position_last).norm();
                        position_last = state.pos_end;
                        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

                        VD(DIM_STATE)
                        K_sum = K.rowwise().sum();
                        VD(DIM_STATE)
                        P_diag = state.cov.diagonal();
                    }
                    EKF_stop_flg = true;
                }
                solve_time += omp_get_wtime() - solve_start;

                if (EKF_stop_flg)
                    break;
            }
        }

#endif
        // SaveTrajTUM(LidarMeasures.lidar_beg_time, state.rot_end, state.pos_end);
        double t_update_end = omp_get_wtime();
        /******* Publish odometry *******/
        euler_cur = RotMtoEuler(state.rot_end);
        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
        publish_odometry(pubOdomAftMapped);

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();
        map_incremental();
        t5 = omp_get_wtime();
        kdtree_incremental_time = t5 - t3 + readd_time;
        /******* Publish points *******/

        PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_pub = *laserCloudWorld;

        if (!img_en)
        {
            publish_frame_world(pubLaserCloudFullRes);
        }
        // publish_visual_world_map(pubVisualCloud);
        publish_effect_world(pubLaserCloudEffect);
        // publish_map(pubLaserCloudMap);
        publish_path(pubPath);
#ifdef DEPLOY
        publish_mavros(mavros_pose_publisher);
#endif

        /*** Debug variables ***/
        frame_num++;
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
#ifdef USE_IKFOM
        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
#else
        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
#endif

        T1[time_log_counter] = LidarMeasures.lidar_beg_time;
        s_plot[time_log_counter] = aver_time_consu; 
        s_plot2[time_log_counter] = kdtree_incremental_time;
        s_plot3[time_log_counter] = kdtree_search_time / kdtree_search_counter;
        s_plot4[time_log_counter] = featsFromMapNum;
        s_plot5[time_log_counter] = t5 - t0;
        time_log_counter++;
        printf("[ LIO ]: time: fov_check: %0.6f fov_check and readd: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f icp: %0.6f construct H: %0.6f.\n", fov_check_time, t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
        if (lidar_en)
        {
            euler_cur = RotMtoEuler(state.rot_end);
#ifdef USE_IKFOM
            fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose() * 57.3 << " " << state_point.pos.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
#else
            fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose() * 57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose()
                     << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << " " << feats_undistort->points.size() << endl;
#endif
        }
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("rgb_scan_all.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current rgb scan saved" << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (pcl_wait_save_lidar->size() > 0 && pcd_save_en)
    {
        string file_name = string("intensity_sacn_all.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current intensity scan saved" << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_lidar);
    }

    fout_out.close();
    fout_pre.close();

#ifndef DEPLOY
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
    FILE *fp2;
    string log_dir = root_dir + "/Log/fast_livo_time_log.csv";
    fp2 = fopen(log_dir.c_str(), "w");
    fprintf(fp2, "time_stamp, average time, incremental time, search time,fov check time, total time, alpha_bal, alpha_del\n");
    for (int i = 0; i < time_log_counter; i++)
    {
        fprintf(fp2, "%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%f,%f\n", T1[i], s_plot[i], s_plot2[i], s_plot3[i], s_plot4[i], s_plot5[i], s_plot6[i], s_plot7[i]);
        t.push_back(T1[i]);
        s_vec.push_back(s_plot[i]);
        s_vec2.push_back(s_plot2[i]);
        s_vec3.push_back(s_plot3[i]);
        s_vec4.push_back(s_plot4[i]);
        s_vec5.push_back(s_plot5[i]);
        s_vec6.push_back(s_plot6[i]);
        s_vec7.push_back(s_plot7[i]);
    }
    fclose(fp2);

#endif
    return 0;
}
