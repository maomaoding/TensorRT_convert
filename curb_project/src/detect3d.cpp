#include "detect3d.h"
static int n=0;
bool comp_down(const pcl::PointXYZ &a, const pcl::PointXYZ &b)
{
	return a.x > b.x;
}

int slideForGettingPoints(pcl::PointCloud<pcl::PointXYZ> &points, pcl::PointCloud<pcl::PointXYZ> &curb_left)
{
	int w_0 = 10;
	int w_d = 10;
	int i = 0;

	// some important parameters influence the final performance.
	float xy_thresh = 0.1;
	float z_thresh = 0.08;

	int points_num = points.size();

	while((i + w_d) < points_num)
	{
		float z_max = points[i].z;
		float z_min = points[i].z;

		int idx_ = 0;
		float z_dis = 0;

		for (int i_ = 0; i_ < w_d; i_++)
		{
			float dis = fabs(points[i+i_].z - points[i+i_+1].z);
			if (dis > z_dis) {z_dis = dis; idx_ = i+i_;}
			if (points[i+i_].z < z_min){z_min = points[i+i_].z;}
			if (points[i+i_].z > z_max){z_max = points[i+i_].z;}
		}

		if (fabs(z_max - z_min) >= z_thresh)
		{
			for (int i_ = 0; i_ < (w_d - 1); i_++)
			{
				float p_dist = sqrt(((points[i + i_].y - points[i + 1 + i_].y) * (points[i + i_].y - points[i + 1 + i_].y)) 
				+ ((points[i + i_].x - points[i + 1 + i_].x) *(points[i + i_].x - points[i + 1 + i_].x)));
				if (p_dist >= xy_thresh)
				{
					curb_left.push_back(points[i_ + i]);
					return 0;
				}
			}
			curb_left.push_back(points[idx_]);
			return 0;
		}
		i += w_0;
	}
}

template<typename T>
std::vector<T> polyfit_Eigen(const std::vector<T> &xValues, const std::vector<T> &yValues, const int degree)
{
	int numCoefficients = degree + 1;
	size_t nCount = xValues.size();

	Eigen::MatrixXf X(nCount, numCoefficients);
	Eigen::MatrixXf Y(nCount, 1);

	for(size_t i = 0; i < nCount; i++)
	{
		Y(i,0) = yValues[i];
	}

	for(size_t nRow = 0; nRow < nCount; nRow++)
	{
		T nVal = 1.0f;
		for(int nCol = 0; nCol < numCoefficients; nCol++)
		{
			X(nRow, nCol) = nVal;
			nVal *= xValues[nRow];
		}
	}

	Eigen::VectorXf coefficients;
	coefficients = X.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
	return std::vector<T>(coefficients.data(), coefficients.data() + numCoefficients);
}

template<typename T>
T polyeval(std::vector<T> &coefficient, T x)
{
	T result = 0.;
	for(int i = 0; i < coefficient.size(); i++)
	{
		result += coefficient[i]*pow(x, i);
	}
	return result;
}


Detect3d::Detect3d(std::string model_path, ros::NodeHandle &nh, std::string pub_name)
{
	model.reset(new TRTmodel("", "", model_path));

	image_transport::ImageTransport it(nh);
	pub_img_ = it.advertise(pub_name, 30);
	pcl_pub= nh.advertise<sensor_msgs::PointCloud2>("cloud_data", 30);

	const char* INPUT_BLOB_NAME = "blob1";
	const char* OUTPUT_BLOB_NAME = "conv_blob203";

	dims_input = model->getTensorDims(INPUT_BLOB_NAME);
	dims_output = model->getTensorDims(OUTPUT_BLOB_NAME);
}

void Detect3d::forward(cv::Mat &frame, pcl::PointCloud<pcl::PointXYZ> ori_cloud,
										Eigen::Quaterniond rotation_visual2baselink,
										Eigen::Vector3d translation_visual2baselink,
										Eigen::Quaterniond rotation_velodyne2baselink,
										Eigen::Vector3d translation_velodyne2baselink)
{
	if(!(frame.cols > 1 && frame.rows > 1))
	{
		return;
	}

	cv::Mat resize_frame;
	cv::resize(frame, resize_frame, cv::Size(dims_input.w(), dims_input.h()));

	float *data;
	float prob[dims_output.w()*dims_output.h()*dims_output.c()];

	float mean[3] = {.485, .456, .406};
	float std_[3] = {.229, .224, .225};

	data = (float*)malloc(dims_input.w()*dims_input.h()*dims_input.c()*sizeof(float));
	for(int c = 0;c < dims_input.c(); c++)
	{
		for(int row = 0;row < dims_input.h();row++)
		{
			for(int col = 0;col < dims_input.w();col++)
			{
				float value = static_cast<float>(resize_frame.at<cv::Vec3b>(row,col)[2-c]);
				data[c*dims_input.h()*dims_input.w()+row*dims_input.w()+col] = (value / 255. - mean[c]) / std_[c];
			}
		}
	}

	model->doInference(data, prob, 1);

	cv::Mat argmax_img = cv::Mat::zeros(resize_frame.size(), CV_8UC3);

	for(int i=0;i<argmax_img.rows;i++)
	{
		for(int j=0;j<argmax_img.cols;j++)
		{
			int hw = argmax_img.rows * argmax_img.cols;
			if(prob[0*hw + i * argmax_img.cols + j] < prob[1*hw + i * argmax_img.cols + j])
			{
				argmax_img.at<cv::Vec3b>(i,j)[0] = 0;
				argmax_img.at<cv::Vec3b>(i,j)[1] = 0;
				argmax_img.at<cv::Vec3b>(i,j)[2] = 255;
			}
		}
	}
	free(data);

	// cv::addWeighted(resize_frame, 1, argmax_img, 1, 0, resize_frame);

	// sensor_msgs::ImagePtr msg_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", resize_frame).toImageMsg();
	// pub_img_.publish(msg_image);



	cv::Mat intrinsics_ = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat distortion_coeff_ = cv::Mat::zeros(5, 1, CV_64F);
	std::string calibration_file = "/home/ubuntu/zhq/tf/perception/6mm.ini";
	std::string camera_name = "camera";
	sensor_msgs::CameraInfo camera_calibration_data;

	camera_calibration_parsers::readCalibrationIni(calibration_file, camera_name, camera_calibration_data);
	for(size_t i = 0; i < 3; i++) {
		for(size_t j = 0; j < 3; j++) {
			intrinsics_.at<double>(i,j) = camera_calibration_data.K.at(3*i+j);
		}
	}
	for(size_t i = 0; i < 5; i++) {
		distortion_coeff_.at<double>(i,0) = camera_calibration_data.D.at(i);
	}

	Eigen::MatrixXd projection_visual2baselink(4, 4);
	Eigen::MatrixXd rotation_tmp(3, 3);
	Eigen::MatrixXd projection_velodyne2baselink(4, 4);
	Eigen::MatrixXd camera_matrix_qc(3, 4);
	Eigen::MatrixXd concat_all(3, 4);
	rotation_tmp = rotation_visual2baselink.toRotationMatrix();
	for(size_t i = 0; i < 3; i++) {
		for(size_t j = 0; j < 3; j++) {
			projection_visual2baselink(i,j) = rotation_tmp(i,j);
		}
	}
	projection_visual2baselink(0,3)=translation_visual2baselink.x();
	projection_visual2baselink(1,3)=translation_visual2baselink.y();
	projection_visual2baselink(2,3)=translation_visual2baselink.z();
	projection_visual2baselink(3,0)=0;
	projection_visual2baselink(3,1)=0;
	projection_visual2baselink(3,2)=0;
	projection_visual2baselink(3,3)=1;

	rotation_tmp = rotation_velodyne2baselink.toRotationMatrix();
	for(size_t i = 0; i < 3; i++) {
		for(size_t j = 0; j < 3; j++) {
			projection_velodyne2baselink(i,j) = rotation_tmp(i,j);
		}
	}
	projection_velodyne2baselink(0,3)=translation_velodyne2baselink.x();
	projection_velodyne2baselink(1,3)=translation_velodyne2baselink.y();
	projection_velodyne2baselink(2,3)=translation_velodyne2baselink.z();
	projection_velodyne2baselink(3,0)=0;
	projection_velodyne2baselink(3,1)=0;
	projection_velodyne2baselink(3,2)=0;
	projection_velodyne2baselink(3,3)=1;

	for(size_t i = 0; i < 3; i++) {
		for(size_t j = 0; j < 3; j++) {
			camera_matrix_qc(i,j) = camera_calibration_data.K.at(3*i+j);
		}
	}
	camera_matrix_qc(0,3)=0;
	camera_matrix_qc(1,3)=0;
	camera_matrix_qc(2,3)=0;

	concat_all=camera_matrix_qc*projection_visual2baselink.inverse();
	double z=-0.5;

	cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create();

	int x,y;

	std::vector<cv::Point> coordinate;
	cv::Mat A;
	int min_x = -1;
	int max_x = -1;               

	for(int i = 0; i < dims_output.h(); i+=1)
	{
		int max_j = -1;
		int max_tmp = 0;
		for(int j = 0; j < dims_output.w(); j++)
		{
			if(prob[1*dims_output.h()*dims_output.w() + i*dims_output.w() + j] -
				prob[0*dims_output.h()*dims_output.w() + i*dims_output.w() + j] >= max_tmp)
			{

				max_j = j;
				max_tmp = prob[1*dims_output.h()*dims_output.w() + i*dims_output.w() + j] -
						prob[0*dims_output.h()*dims_output.w() + i*dims_output.w() + j];
			}
		}
		if(max_j != -1)
			coordinate.push_back(cv::Point(max_j, i));
	}

	cv::polylines(resize_frame, coordinate, false, cv::Scalar(0, 0, 255), 1, 8, 0);

	sensor_msgs::ImagePtr msg_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", resize_frame).toImageMsg();
	pub_img_.publish(msg_image);

	for(int i = 0; i < coordinate.size(); i++)
	{
		x = coordinate[i].x/512.0*1280;
		y = coordinate[i].y/256.0*720;

		cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<DistortF>(x, y, intrinsics_,
																				distortion_coeff_);
		int lFaliedUndistCount=0;
		cv::Mat solution = (cv::Mat_<double>(1, 2) << x, y);
		cv::Mat step = (cv::Mat_<double>(2, 1) << -0.5, -0.5);
		solver->setFunction(ptr_F);
		solver->setInitStep(step);
		float res=solver->minimize(solution);
		double u_not = solution.at<double>(0, 0);
		double v_not = solution.at<double>(0, 1);

		double matrix[3][4];
		matrix[0][0]=concat_all(0,0);
		matrix[0][1]=concat_all(0,1);
		matrix[0][2]=-u_not;
		matrix[0][3]=-(concat_all(0,2)*z+concat_all(0,3));
		matrix[1][0]=concat_all(1,0);
		matrix[1][1]=concat_all(1,1);
		matrix[1][2]=-v_not;
		matrix[1][3]=-(concat_all(1,2)*z+concat_all(1,3));
		matrix[2][0]=concat_all(2,0);
		matrix[2][1]=concat_all(2,1);
		matrix[2][2]=-1;
		matrix[2][3]=-(concat_all(2,2)*z+concat_all(2,3));

		pcl::PointXYZ baselink_point;
		baselink_point = calc<double>(matrix);
		baselink_point.z=z;

		Eigen::Vector4d baselink_point_eigen(baselink_point.x, baselink_point.y, baselink_point.z, 1);
		Eigen::Vector4d velodyne_point_eigen = projection_velodyne2baselink.inverse() * baselink_point_eigen;

		pcl::PointXYZRGB colorcurb;
		colorcurb.x = velodyne_point_eigen(0);
		colorcurb.y = velodyne_point_eigen(1);
		colorcurb.z = velodyne_point_eigen(2);
		colorcurb.r = 255;
		colorcurb.g = 255;
		colorcurb.b = 255;

		curb.push_back(colorcurb);
	}

	//store roi voxel image
	if(curb.points.size() != 0)
	{
		double x_max = curb.points[0].x;
		double x_min = curb.points[0].x;
		double y_max = curb.points[0].y;
		double y_min = curb.points[0].y;
		for(size_t i = 0; i < curb.points.size(); i++)
		{
			x_max = curb.points[i].x > x_max ? curb.points[i].x : x_max;
			x_min = curb.points[i].x < x_min ? curb.points[i].x : x_min;
			y_max = curb.points[i].y > y_max ? curb.points[i].y : y_max;
			y_min = curb.points[i].y < y_min ? curb.points[i].y : y_min;
		}

		x_min -= 0.15;
		// x_max += 0.35;

		curb.clear();
		//add partial original points
		//(-26,-24) (-20,-18) (-14,-13) (-13,-12) (-12,-11.5) (-11.5,-11) (-11,-10) (-10,-9) (-9,-8) (-8,-7)
		float each_line_deg[20] = {-26,-24,-20,-18,-14,-13,-13,-12,-12,-11.5,-11.5,-11,-11,-10,-10,-9,-9,-8,-8,-7};
		std::vector<pcl::PointCloud<pcl::PointXYZ> > pointcloud_10line(10);
		for(size_t i = 0; i < ori_cloud.size(); i++)
		{
			if(ori_cloud.points[i].x > x_min && ori_cloud.points[i].x < x_max
				&& ori_cloud.points[i].y < -1)
			{
				pcl::PointXYZRGB tmp;
				tmp.x = ori_cloud.points[i].x;
				tmp.y = ori_cloud.points[i].y;
				tmp.z = ori_cloud.points[i].z;
				tmp.r = 255;
				tmp.g = 0;
				tmp.b = 0;
				curb.push_back(tmp);

				float deg = atan2(ori_cloud.points[i].z,
					sqrt(ori_cloud.points[i].x*ori_cloud.points[i].x+ori_cloud.points[i].y*ori_cloud.points[i].y))
					*180.0/3.141592653;

				for(size_t i_ = 0; i_ < 10; i_++)
				{
					if(deg > each_line_deg[2*i_] && deg < each_line_deg[2*i_ + 1])
					{
						pointcloud_10line[i_].push_back(ori_cloud.points[i]);
					}
				}
			}
		}

		pcl::PointCloud<pcl::PointXYZ> tmp_curb_point;
		std::vector<float> xvalue, yvalue, coefficient;
		for(size_t i = 0; i < 10; i++)
		{
			sort(pointcloud_10line[i].begin(), pointcloud_10line[i].end(), comp_down);
			slideForGettingPoints(pointcloud_10line[i], tmp_curb_point);
		}
		for(size_t i = 0; i < tmp_curb_point.size(); i++)
		{
			xvalue.push_back(tmp_curb_point[i].x);
			yvalue.push_back(tmp_curb_point[i].y);
		}
		
		if(xvalue.size() >= 2)
		{
			coefficient = polyfit_Eigen<float>(yvalue, xvalue, 3);
			for(float i = yvalue[yvalue.size()-1]; i < 0; i+=0.01)
			{
				float j = polyeval(coefficient, i);
				pcl::PointXYZRGB tmp;
				tmp.x = j;
				tmp.y = i;
				tmp.z = tmp_curb_point[0].z;
				tmp.r = 255;
				tmp.g = 255;
				tmp.b = 255;
				curb.push_back(tmp);
			}
			for(int i = 0; i < yvalue.size(); i++)
			{
				pcl::PointXYZRGB tmp;
				tmp.x = xvalue[i];
				tmp.y = yvalue[i];
				tmp.z = tmp_curb_point[0].z;
				tmp.r = 0;
				tmp.g = 255;
				tmp.b = 0;
				curb.push_back(tmp);
			}
		}
	}

	//store roi voxel image

	sensor_msgs::PointCloud2 curb_out;
	pcl::toROSMsg(curb,curb_out);
	curb_out.header.frame_id = "velodyne";
	pcl_pub.publish(curb_out);

	n++;
	// if(n%30==0)
		curb.clear();
}