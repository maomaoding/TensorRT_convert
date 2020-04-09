#include "camera_base.h"

CamBase::CamBase(ros::NodeHandle &nh)
{
	tf_ = new tf::TransformListener();
	translation_visual2baselink = Eigen::Vector3d(0,0,0);
	rotation_visual2baselink      = Eigen::Quaterniond::Identity();
	translation_velodyne2baselink = Eigen::Vector3d(0,0,0);
	rotation_velodyne2baselink = Eigen::Quaterniond::Identity();

	std::string image_topic, cloud_topic;
	nh.param<std::string>("image_topic", image_topic, "/camera/image_raw");
	// nh.param<std::string>("cloud_topic", cloud_topic, "/main/velodyne_points");
	nh.param<std::string>("cloud_topic", cloud_topic, "/lidar_points");

	image_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, image_topic, 30));
	ori_pointcloud_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, cloud_topic, 30));

	sync_image_and_cloud_.reset(new message_filters::Synchronizer<SyncPolicyImageAndPointcloud>(
							SyncPolicyImageAndPointcloud(30), *image_sub_, *ori_pointcloud_sub_));
	sync_image_and_cloud_->registerCallback(boost::bind(&CamBase::imageCb, this, _1, _2));
}

CamBase::~CamBase()
{
	if(tf_)
		delete tf_;
}

void CamBase::set_net(std::shared_ptr<Detect3d> net)
{
	detect = net;
}

void CamBase::imageCb(const sensor_msgs::ImageConstPtr& msg_image, const sensor_msgs::PointCloud2ConstPtr& msg_cloud)
{
	try
	{
		if(tf_)
		{
			tf::StampedTransform transform;
			tf_->lookupTransform("base_link","visual",ros::Time(0), transform);

			double roll, yaw, pitch;
			auto q = transform.getRotation();
			tf::Matrix3x3 m(q);
			m.getRPY(roll,pitch,yaw);

			ROS_INFO("TF: %f %f %f %f %f %f", transform.getOrigin().x(), 
						transform.getOrigin().y(), 
						transform.getOrigin().z(),
						roll, yaw, pitch);

			translation_visual2baselink.x() = transform.getOrigin().x();
			translation_visual2baselink.y() = transform.getOrigin().y();
			translation_visual2baselink.z() = transform.getOrigin().z();
			rotation_visual2baselink = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
									* Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
									* Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

			tf_->lookupTransform("base_link","velodyne",ros::Time(0), transform);
			q = transform.getRotation();
			tf::Matrix3x3 m_velodyne(q);
			m_velodyne.getRPY(roll,pitch,yaw);
			translation_velodyne2baselink.x() = transform.getOrigin().x();
			translation_velodyne2baselink.y() = transform.getOrigin().y();
			translation_velodyne2baselink.z() = transform.getOrigin().z();
			rotation_velodyne2baselink = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
										* Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
										* Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

			delete tf_;
			tf_ = nullptr;
		}
	}
	catch(tf::TransformException &e)
	{
		ROS_ERROR("TF error: %s", e.what());
		return;
	}

	cv_bridge::CvImagePtr cv_ptr_ = cv_bridge::toCvCopy(msg_image, sensor_msgs::image_encodings::BGR8);
	cv::Mat origin_img_ = cv_ptr_->image;

	pcl::PointCloud<pcl::PointXYZ> ori_cloud;
	pcl::fromROSMsg(*msg_cloud, ori_cloud);

	// string image_save_path = "/home/ubuntu/dyh/tensorrt_trial/curb_voxel/" + 
	// 						to_string(msg_cloud->header.stamp.nsec) + ".jpg";
	// string cloud_save_path = "/home/ubuntu/dyh/tensorrt_trial/curb_voxel/" +
	// 						to_string(msg_cloud->header.stamp.nsec) + ".pcd";
	// cv::imwrite(image_save_path, origin_img_);
	// pcl::io::savePCDFileASCII(cloud_save_path, ori_cloud);

	detect->forward(origin_img_, ori_cloud, rotation_visual2baselink, translation_visual2baselink,
									rotation_velodyne2baselink, translation_velodyne2baselink);
}