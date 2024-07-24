# FAST-LIVO 教程（建设中……）

## 代码梳理

### 原文中相关参数的介绍

Edit `config/xxx.yaml` to set the below parameters:

- `lid_topic`: The topic name of LiDAR data.
- `imu_topic`: The topic name of IMU data.
- `img_topic`: The topic name of camera data.
- `img_enable`: Enbale vio submodule.
- `lidar_enable`: Enbale lio submodule.
- `point_filter_num`: The sampling interval for a new scan. It is recommended that `3~4` for faster odometry, and `1~2` for denser map.
- `outlier_threshold`: The outlier threshold value of photometric error (square) of a single pixel. It is recommended that `50~250` for the darker scenes, and `500~1000` for the brighter scenes. The smaller the value is, the faster the vio submodule is, but the weaker the anti-degradation ability is.
- `img_point_cov`: The covariance of photometric errors per pixel. 
- `laser_point_cov`: The covariance of point-to-plane redisual per point. 
- `filter_size_surf`: Downsample the points in a new scan. It is recommended that `0.05~0.15` for indoor scenes, `0.3~0.5` for outdoor scenes.
- `filter_size_map`: Downsample the points in LiDAR global map. It is recommended that `0.15~0.3` for indoor scenes, `0.4~0.5` for outdoor scenes.
- `pcd_save_en`: If `true`, save point clouds to the PCD folder. Save RGB-colored points if `img_enable` is `1`, intensity-colored points if `img_enable` is `0`.
- `delta_time`: The time offset between the camera and LiDAR, which is used to correct timestamp misalignment.
