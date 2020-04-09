#ifndef DETECT3D_
#define	DETECT3D_
#include "utils.h"
#include "trtmodel.h"
#include <Eigen/Dense>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <camera_calibration_parsers/parse_ini.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/surface/on_nurbs/fitting_curve_2d.h>
// #include <pcl/surface/on_nurbs/triangulation.h>

class Detect3d
{
public:
	Detect3d(std::string model_path, ros::NodeHandle &nh, std::string pub_name);
	// ~Detect3d();
	void forward(cv::Mat &frame, pcl::PointCloud<pcl::PointXYZ> ori_cloud, 
								Eigen::Quaterniond rotation_visual2baselink,
								Eigen::Vector3d translation_visual2baselink,
								Eigen::Quaterniond rotation_velodyne2baselink,
								Eigen::Vector3d translation_velodyne2baselink);

private:
	std::shared_ptr<TRTmodel> model;

	nvinfer1::DimsCHW dims_input;
	nvinfer1::DimsCHW dims_output;

	image_transport::Publisher pub_img_;
	ros::Publisher pcl_pub;
	ros::Publisher pcl_pub_enu;
	pcl::PointCloud<pcl::PointXYZRGB> curb;
};

//------------------------------------DistortF:Get points without distortion x y z_c--------------------------------------------------------------------------------
class DistortF :public cv::MinProblemSolver::Function
{
public:
	DistortF(double _distortU, double _distortV, cv::Mat intrinsics_, cv::Mat distortion_coeff_) : distortU(_distortU), distortV(_distortV) ,cameraMatrix(intrinsics_)
	{

		inverseCameraMatrix = (cameraMatrix.colRange(0,3)).inv(cv::DECOMP_LU);
		k1 = distortion_coeff_.at<double>(0,0);
		k2 = distortion_coeff_.at<double>(1,0);
		p1 = distortion_coeff_.at<double>(2,0);
		p2 = distortion_coeff_.at<double>(3,0);
		k3 = distortion_coeff_.at<double>(4,0);
		k4 =  0.;
		k5 = 0.;
		k6 = 0.;
		s1 = 0.;
		s2 =  0.;
		s3 = 0.;
		s4 = 0.;
		tauX =  0.;
		tauY =  0.;
	}
	int getDims() const { return 2; }
	double calc(const double* x)const{

		double undistortU = x[0];
		double undistortV = x[1];
		const double* ir = &inverseCameraMatrix(0, 0);
		//cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1953.701994, 0.000000, 673.774516 ,0.,1958.362344, 286.445366, 0., 0., 1.);
		const double* pA = (const double*)cameraMatrix.data;

		double Xd = undistortU * ir[0] + undistortV * ir[1] + ir[2], Yd = undistortU * ir[3] + undistortV * ir[4] + ir[5], Wd = undistortU * ir[6] + undistortV * ir[7] + ir[8];
		Wd = 1. / Wd;
		Xd = Xd * Wd;
		Yd = Yd * Wd;

		double Xd_2 = Xd*Xd, Yd_2 = Yd * Yd, r_2 = Xd_2 + Yd_2, _2XdYd = 2 * Xd * Yd;
		double kr = (1 + ((k3*r_2 + k2)*r_2 + k1)*r_2) / (1 + ((k6*r_2 + k5)*r_2 + k4)*r_2);
		double Xdd = (Xd*kr + p1*_2XdYd + p2*(r_2 + 2 * Xd_2) + s1*r_2 + s2*r_2*r_2);;
		double Ydd = (Yd*kr + p1*(r_2 + 2 * Yd_2) + p2*_2XdYd + s3*r_2 + s4*r_2*r_2);
		double Wdd = Wd;

		double distortU_d = pA[0] * Xdd + pA[1] * Ydd + pA[2] * Wdd;
		double distortV_d = pA[3] * Xdd + pA[4] * Ydd + pA[5] * Wdd;
		//printf("%f\n",sqrt((distortU - distortU_d) * (distortU - distortU_d) + (distortV - distortV_d) * (distortV - distortV_d)));
		return sqrt((distortU - distortU_d) * (distortU - distortU_d) + (distortV - distortV_d) * (distortV - distortV_d));
	}
private:
	double distortU, distortV;
	cv::Mat cameraMatrix;   
	cv::Mat distortion_coeff_ ;
	double k1;
	double k2;
	double p1;
	double p2;
	double k3;
	double k4;
	double k5;
	double k6;
	double s1;
	double s2;
	double s3;
	double s4;
	double tauX;
	double tauY;
	cv::Mat_<double> inverseCameraMatrix;
};

//-----------------------------------------Determinant:Get x,y,z_c--------------------------------------------------------------------------------
template<class T>
pcl::PointXYZ calc(T matrix[3][4]){
    T     base_D = matrix[0][0]*matrix[1][1]*matrix[2][2] + matrix[1][0]*matrix[2][1]*matrix[0][2] + matrix[2][0]*matrix[0][1]*matrix[1][2];
    base_D = base_D-(matrix[0][2]*matrix[1][1]*matrix[2][0] + matrix[0][0]*matrix[1][2]*matrix[2][1] + matrix[0][1]*matrix[1][0]*matrix[2][2]);

    if(base_D != 0){
        T     x_D = matrix[0][3]*matrix[1][1]*matrix[2][2] + matrix[1][3]*matrix[2][1]*matrix[0][2] + matrix[2][3]*matrix[0][1]*matrix[1][2];
        x_D = x_D-(matrix[0][2]*matrix[1][1]*matrix[2][3] + matrix[0][3]*matrix[1][2]*matrix[2][1] + matrix[0][1]*matrix[1][3]*matrix[2][2]);
        T     y_D = matrix[0][0]*matrix[1][3]*matrix[2][2] + matrix[1][0]*matrix[2][3]*matrix[0][2] + matrix[2][0]*matrix[0][3]*matrix[1][2];
        y_D = y_D-(matrix[0][2]*matrix[1][3]*matrix[2][0] + matrix[0][0]*matrix[1][2]*matrix[2][3] + matrix[0][3]*matrix[1][0]*matrix[2][2]);
        T     z_D = matrix[0][0]*matrix[1][1]*matrix[2][3] + matrix[1][0]*matrix[2][1]*matrix[0][3] + matrix[2][0]*matrix[0][1]*matrix[1][3];
        z_D = z_D-(matrix[0][3]*matrix[1][1]*matrix[2][0] + matrix[0][0]*matrix[1][3]*matrix[2][1] + matrix[0][1]*matrix[1][0]*matrix[2][3]);

        T x =  x_D/base_D;
        T y =  y_D/base_D;
        T z =  z_D/base_D;
        pcl::PointXYZ tmp;
        tmp.x=x;
        tmp.y=y;
        tmp.z=z;
        return tmp;
    }
}

#endif