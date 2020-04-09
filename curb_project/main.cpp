#include "camera_base.h"

static std::shared_ptr<Detect3d> CreateModel(ros::NodeHandle &nh, std::string model)
{  
	std::string path, output;
	nh.param<std::string>("/"+model + "_model", path, "");
	nh.param<std::string>(model + "_debug", output, model);

	if (path == "")
		return nullptr;
	return std::shared_ptr<Detect3d>(new Detect3d(path, nh, output));
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "curb_perception");
	ros::NodeHandle nh("~");

	std::string net_name = "curb";
	ROS_INFO("load %s", net_name.c_str());
	std::shared_ptr<Detect3d> net = CreateModel(nh, net_name);

	std::shared_ptr<CamBase> ptr(new CamBase(nh));
	ptr->set_net(net);

	ROS_INFO("load models done");

	ros::spin();
	return 0;
}