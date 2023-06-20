/**
 * @file LoopOptimization.hpp
 * @author your name (you@domain.com)
 * @brief 用于进行回环优化的部分
 * @version 0.1
 * @date 2023-06-17
 * 
 * @copyright Copyright (c) 2023
 * @todo 
 * 1. 构建关键帧（完成）
 * 2. 将关键帧添加到因子图中（完成）
 * 3. 检测回环优化(完成)
 * 4. 回环优化后对姿态和局部地图进行重新矫正
 */
#include <ros/ros.h> // 用于发布关键帧可视化
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <Eigen/Eigen>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <iostream>
#include <mutex>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class LoopOptimization
{
public:
    // 定义关键帧的数据
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D_; // 关键帧的位置
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D_; // 关键帧的位置
    std::vector<pcl::PointCloud<PointType>::Ptr> pointCloud_keyFrame_; // 关键帧的点云（lidar系下）
    std::vector<float *> transformTobeMapped_vector_; // 关键帧的姿态六维
    bool transformTobeMapped_update_flag_;
    int KeyFrameNumber;
    double timeLaserInfoCur_;
    pcl::VoxelGrid<PointType> downSizeFilterICP;

    // 回环信息数据
    std::map<int, int> loopIndexContainer_; // from new to old
    std::vector<std::pair<int, int>> loopIndexQueue_;
    std::vector<gtsam::Pose3> loopPoseQueue_;
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue_;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses_;

    // 关键帧参数
    float surroundingkeyframeAddingDistThreshold_;
    float surroundingkeyframeAddingAngleThreshold_;

    // Loop closure
    bool aLoopIsClosed_;
    bool  loopClosureEnableFlag_;
    float loopClosureFrequency_;
    int   surroundingKeyframeSize_;
    float historyKeyframeSearchRadius_;
    float historyKeyframeSearchTimeDiff_;
    int   historyKeyframeSearchNum_;
    float historyKeyframeFitnessScore_;
    string odometryFrame_;

    ros::Publisher pubLoopConstraintEdge_;

    std::mutex mtxLoopInfo;

    // voxel filter paprams
    float pointCloudLeafSize_;

    // gtsam
    gtsam::NonlinearFactorGraph gtSAMgraph_;
    gtsam::Values initialEstimate_;
    gtsam::Values optimizedEstimate_;
    gtsam::ISAM2 *isam_;
    gtsam::Values isamCurrentEstimate_;

    
public:
    LoopOptimization(const ros::Publisher & pubLoopConstraint);
    ~LoopOptimization();
    bool saveKeyFrame(pcl::PointCloud<PointType>::Ptr cloud_input, float pose[6]);
    void addOdomFactor(pcl::PointCloud<PointType>::Ptr cloud_input, float pose[6], double lidar_time);
    void addLoopFactor();
    void optimization();
    void gtsamClear();

    void publish_KeyFrame(const ros::Publisher & pubOdomAftMapped, double lidar_time, string odometryFrame);
    
    void loopClosureThread();
    void visualizeLoopClosure();
    bool detectLoopClosureDistance(int *latestID, int *closestID);
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum);
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, float * transformIn);

    void correctPoses();
    void getCurrpose(float * transformOut);
    // 位姿态变换函数
    gtsam::Pose3 trans2gtsamPose(float transformIn[]);
    Eigen::Affine3f trans2Affine3f(float transformIn[]);
};

LoopOptimization::LoopOptimization(const ros::Publisher & pubLoopConstraint)
{
    // 对数据进行初始化
    cloudKeyPoses3D_.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses3D_.reset(new pcl::PointCloud<PointType>());

    // 参数初始化，后期使用yaml文件调试参数
    KeyFrameNumber = 0;
    surroundingkeyframeAddingDistThreshold_ = 1.0;
    surroundingkeyframeAddingAngleThreshold_ = 0.2;
    aLoopIsClosed_ = false;

    // ISM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam_ = new gtsam::ISAM2(parameters);

    // 发布激光里程计，rviz中表现为坐标轴

    loopClosureEnableFlag_ = true;
    loopClosureFrequency_ = 1.0;
    historyKeyframeSearchNum_ = 20;

    historyKeyframeSearchRadius_ = 5.0;

    kdtreeHistoryKeyPoses_.reset(new pcl::KdTreeFLANN<PointType>());
    timeLaserInfoCur_ = 1.0;
    historyKeyframeSearchTimeDiff_ = 40.0;

    downSizeFilterICP.setLeafSize(0.4, 0.4, 0.4);

    historyKeyframeFitnessScore_ = 0.3;

    pubLoopConstraintEdge_ = pubLoopConstraint;

    transformTobeMapped_update_flag_ = false;
    std::cout << "LoopOptimization init succeed!" << std::endl;
}

LoopOptimization::~LoopOptimization()
{
}

// 输入本帧的点云和姿态，构建关键帧 PointCloudXYZI::Ptr
bool LoopOptimization::saveKeyFrame(pcl::PointCloud<PointType>::Ptr cloud_input, float pose[6]){
    // 为第一帧创建关键帧
    if (cloudKeyPoses3D_->points.empty())
        return true;
    // 前一帧位姿
    Eigen::Affine3f transStart = trans2Affine3f(transformTobeMapped_vector_.back());
    // 当前帧位姿
    Eigen::Affine3f transFinal = trans2Affine3f(pose);
    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll)  < surroundingkeyframeAddingAngleThreshold_ &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold_ && 
        abs(yaw)   < surroundingkeyframeAddingAngleThreshold_ &&
        sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold_)
        return false;

    return true;
}


// 添加新关键帧，添加到gtsam优化中去
void LoopOptimization::addOdomFactor(pcl::PointCloud<PointType>::Ptr cloud_input, float pose[6], double lidar_time){
    // 添加到因子图中
    if (cloudKeyPoses3D_->points.empty()){
        // 第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(pose), priorNoise));
        // 变量节点设置初始值
        initialEstimate_.insert(0, trans2gtsamPose(pose));
    } else {
        // 添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = trans2gtsamPose(transformTobeMapped_vector_.back());
        gtsam::Pose3 poseTo   = trans2gtsamPose(pose);
        // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D_->size()-1, cloudKeyPoses3D_->size(), poseFrom.between(poseTo), odometryNoise));
        // 变量节点设置初始值
        initialEstimate_.insert(cloudKeyPoses3D_->size(), poseTo);
    }
    // 添加关键帧信息
    pcl::PointCloud<PointType>::Ptr thisKeyFrame_PointCloud(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*cloud_input, *thisKeyFrame_PointCloud);
    pointCloud_keyFrame_.push_back(thisKeyFrame_PointCloud); 
    // (下面这些，应该在优化结束后添加)
    float* array = new float[6];  // 使用 new 运算符在堆区分配内存(注意内存的管理)
    for(int i = 0; i < 6; i++)
        array[i] = pose[i];
    transformTobeMapped_vector_.push_back(array); // 这样的数据保存有问题
    PointType pose3D;
    pose3D.x = pose[3];
    pose3D.y = pose[4];
    pose3D.z = pose[5];
    pose3D.intensity = float(KeyFrameNumber); // 记录当前点对应的帧的id
    pose3D.normal_x = float(lidar_time); // 用normal_x记录时间
    cloudKeyPoses3D_->points.push_back(pose3D);
    timeLaserInfoCur_ = lidar_time;
    KeyFrameNumber++;
}

// 添加闭环因子(这里需要外部的数据)
void LoopOptimization::addLoopFactor(){
    if (loopIndexQueue_.empty())
            return;
    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue_.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue_[i].first;
        int indexTo = loopIndexQueue_[i].second;
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue_[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue_[i];
        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }
    std::cout << "----------插入回环优化了-----------" << std::endl;

    loopIndexQueue_.clear();
    loopPoseQueue_.clear();
    loopNoiseQueue_.clear();
    aLoopIsClosed_ = true;
}

void LoopOptimization::optimization(){
    // gtSAMgraph_.print("--------------------------\n");
    // 执行优化
    isam_->update(gtSAMgraph_, initialEstimate_);
    isam_->update();
}

void LoopOptimization::gtsamClear(){
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph_.resize(0);
    initialEstimate_.clear();
}

// 发布一个坐标轴，作为关键帧的位姿
void LoopOptimization::publish_KeyFrame(const ros::Publisher & pubOdomAftMapped, double lidar_time, string odometryFrame){
    // 发布激光里程计，odom等价map
    float *transform = transformTobeMapped_vector_.back(); // 指向同一个堆区
    odometryFrame_ = odometryFrame;
    nav_msgs::Odometry laserKeyFrameROS;
    laserKeyFrameROS.header.stamp = ros::Time().fromSec(lidar_time);
    laserKeyFrameROS.header.frame_id = odometryFrame;
    laserKeyFrameROS.child_frame_id = "KeyFrame";
    laserKeyFrameROS.pose.pose.position.x = transform[3];
    laserKeyFrameROS.pose.pose.position.y = transform[4];
    laserKeyFrameROS.pose.pose.position.z = transform[5];
    laserKeyFrameROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transform[0], transform[1], transform[2]);
    pubOdomAftMapped.publish(laserKeyFrameROS);
}



// 进行欧氏距离的回环检测函数
/**
 * 闭环线程
 * 1、闭环scan-to-map，icp优化位姿
 *   1) 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
 *   2) 提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
 *   3) 执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
 * 2、rviz展示闭环边
*/
void LoopOptimization::loopClosureThread()
{
    if (loopClosureEnableFlag_ == false)
        return;

    ros::Rate rate(loopClosureFrequency_);
    while (ros::ok())
    {
        rate.sleep();
        // 闭环scan-to-map，icp优化位姿
        // 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
        // 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        // 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
        // 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
        if (cloudKeyPoses3D_->points.empty() == true)
            continue; // 没有直接跳过
        mtxLoopInfo.lock();
        *copy_cloudKeyPoses3D_ = *cloudKeyPoses3D_; // 取出位置点
        mtxLoopInfo.unlock();

        // 当前关键帧索引，候选闭环匹配帧索引
        int loopKeyCur;
        int loopKeyPre;

        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
            continue;
        std::cout <<  "\033[1;31m" << "find loop succed!"  << "\033[0m" << std::endl;
        // 提取
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        // 提取当前关键帧特征点集合，降采样
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum_);

        // 如果特征点较少，结束
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            continue;
        // ICP参数设置
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius_*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // scan-to-map，调用icp匹配
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // 未收敛，或者匹配不够好
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore_)
            continue;
        std::cout <<  "\033[1;31m" << "the loop is useful ! the Score is : "  << "\033[0m" << icp.getFitnessScore() <<  std::endl;

        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云


        // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = trans2Affine3f(transformTobeMapped_vector_[copy_cloudKeyPoses3D_->points[loopKeyCur].intensity]);
        // 闭环优化后当前帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));

        // 闭环匹配帧的位姿
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped_vector_[copy_cloudKeyPoses3D_->points[loopKeyPre].intensity]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

        // 添加闭环因子需要的数据
        mtxLoopInfo.lock();
        loopIndexQueue_.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue_.push_back(poseFrom.between(poseTo));
        loopNoiseQueue_.push_back(constraintNoise);
        mtxLoopInfo.unlock();

        loopIndexContainer_[loopKeyCur] = loopKeyPre;

        // rviz展示闭环边
        visualizeLoopClosure();
    }
}

// rviz展示闭环边
void LoopOptimization::visualizeLoopClosure(){
    if (loopIndexContainer_.empty())
        return;
    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame_;
    markerNode.header.stamp = ros::Time().fromSec(cloudKeyPoses3D_->points.back().normal_x);
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;

    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame_;
    markerEdge.header.stamp = ros::Time().fromSec(cloudKeyPoses3D_->points.back().normal_x);
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;
    // 遍历闭环
    for (auto it = loopIndexContainer_.begin(); it != loopIndexContainer_.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = cloudKeyPoses3D_->points[key_cur].x;
        p.y = cloudKeyPoses3D_->points[key_cur].y;
        p.z = cloudKeyPoses3D_->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = cloudKeyPoses3D_->points[key_pre].x;
        p.y = cloudKeyPoses3D_->points[key_pre].y;
        p.z = cloudKeyPoses3D_->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge_.publish(markerArray);
}

bool LoopOptimization::detectLoopClosureDistance(int *latestID, int *closestID){
    // 当前关键帧帧
    int loopKeyCur = copy_cloudKeyPoses3D_->size() - 1;
    int loopKeyPre = -1;

    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer_.find(loopKeyCur);
    if (it != loopIndexContainer_.end())
        return false;
    
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses_->setInputCloud(copy_cloudKeyPoses3D_);
    kdtreeHistoryKeyPoses_->radiusSearch(copy_cloudKeyPoses3D_->back(), historyKeyframeSearchRadius_, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses3D_->points[id].normal_x - timeLaserInfoCur_) > historyKeyframeSearchTimeDiff_)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}


/**
 * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
*/
void LoopOptimization::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses3D_->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize )
            continue;
        *nearKeyframes += *transformPointCloud(pointCloud_keyFrame_[keyNear], transformTobeMapped_vector_[copy_cloudKeyPoses3D_->points[keyNear].intensity]);
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

/**
 * 对点云cloudIn进行变换transformIn，返回结果点云
*/
pcl::PointCloud<PointType>::Ptr LoopOptimization::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, float * transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    
    // 多线程加速(暂时不开启)
    // #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

// 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹(对正常输出进行一帧的输出，检测到回环重新全部矫正)
// 还有对局部地图进行变换（在检测到回环后进行）
void LoopOptimization::correctPoses(){
    if(!aLoopIsClosed_){ // 没有回环只得到最近一帧的位置和姿态
        // gtsam::Pose3 latestEstimate;
        // isamCurrentEstimate_ = isam_->calculateEstimate();
        // latestEstimate = isamCurrentEstimate_.at<gtsam::Pose3>(isamCurrentEstimate_.size()-1); // 最新一帧
        // // 添加新的优化后的位置
        // float* array = new float[6];  // 使用 new 运算符在堆区分配内存(注意内存的管理)
        // array[0] = latestEstimate.rotation().roll();
        // array[1] = latestEstimate.rotation().pitch();
        // array[2] = latestEstimate.rotation().yaw();
        // array[3] = latestEstimate.translation().x();
        // array[4] = latestEstimate.translation().y();
        // array[5] = latestEstimate.translation().z();
        // transformTobeMapped_vector_.push_back(array); // 这样的数据保存有问题
        // PointType pose3D;
        // pose3D.x = array[3];
        // pose3D.y = array[4];
        // pose3D.z = array[5];
        // pose3D.intensity = float(KeyFrameNumber); // 记录当前点对应的帧的id
        // pose3D.normal_x = float(timeLaserInfoCur_); // 用normal_x记录时间
        // cloudKeyPoses3D_->points.push_back(pose3D);
        // 这里暂时就不对协方差进行更新了
    } else { // 对所有位置都进行修改
        aLoopIsClosed_ = false;
        if (cloudKeyPoses3D_->points.empty())
            return;
        // 清空局部map

        // 清空里程计轨迹

        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        isamCurrentEstimate_ = isam_->calculateEstimate();
        int numPoses = isamCurrentEstimate_.size();
        for (int i = 0; i < numPoses; ++i) {
            cloudKeyPoses3D_->points[i].x = isamCurrentEstimate_.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D_->points[i].y = isamCurrentEstimate_.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D_->points[i].z = isamCurrentEstimate_.at<gtsam::Pose3>(i).translation().z();

            transformTobeMapped_vector_[i][0] = isamCurrentEstimate_.at<gtsam::Pose3>(i).rotation().roll();
            transformTobeMapped_vector_[i][1] = isamCurrentEstimate_.at<gtsam::Pose3>(i).rotation().pitch();
            transformTobeMapped_vector_[i][2] = isamCurrentEstimate_.at<gtsam::Pose3>(i).rotation().yaw();
            transformTobeMapped_vector_[i][3] = cloudKeyPoses3D_->points[i].x;
            transformTobeMapped_vector_[i][4] = cloudKeyPoses3D_->points[i].y;
            transformTobeMapped_vector_[i][5] = cloudKeyPoses3D_->points[i].z;
            
            // 更新里程计轨迹(尽量不要在里面进行显示，发送到主函数让他显示)
            // updatePath(transformTobeMapped_vector_[i]);
        }
        std::cout << "回环优化后的位姿态重置完毕！！" << std::endl;
        transformTobeMapped_update_flag_ = true; // 告诉外部程序，这里已经进行了更新了，需要重新发布一次整个的状态
    }
}

void LoopOptimization::getCurrpose(float * transformOut){
    for(int i = 0; i < 6; i++)
        transformOut[i] = transformTobeMapped_vector_.back()[i];
}

/**
 * 位姿格式变换
*/
gtsam::Pose3 LoopOptimization::trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

/**
 * Eigen格式的位姿变换
*/
Eigen::Affine3f LoopOptimization::trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], 
                                    transformIn[0], transformIn[1], transformIn[2]);
}

