#ifndef CAMERA_BASE_
#define CAMERA_BASE_
#include "utils.h"
#include "detect3d.h"
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>

class CamBase
{
public:
	CamBase(ros::NodeHandle &nh);
	~CamBase();
	void imageCb(const sensor_msgs::ImageConstPtr& msg_image, const sensor_msgs::PointCloud2ConstPtr& msg_cloud);
	void set_net(std::shared_ptr<Detect3d> net);
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
	std::shared_ptr<Detect3d> detect = nullptr;
	tf::TransformListener* tf_ = nullptr;

	Eigen::Quaterniond rotation_visual2baselink;
	Eigen::VectorXd translation_visual2baselink;
	Eigen::Quaterniond rotation_velodyne2baselink;
	Eigen::VectorXd translation_velodyne2baselink;

	std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image> > image_sub_;
	std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2> > ori_pointcloud_sub_;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::PointCloud2>
																			SyncPolicyImageAndPointcloud;
	std::unique_ptr<message_filters::Synchronizer<SyncPolicyImageAndPointcloud> > sync_image_and_cloud_;
};

#endif