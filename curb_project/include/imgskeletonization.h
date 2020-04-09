#ifndef SKELETONIZATION_HPP_
#define SKELETONIZATION_HPP_

#include <opencv2/opencv.hpp>
using namespace cv;

Mat ImgSkeletonization(Mat &input_src,Mat & output_dst, int number);
Mat ImgSkeletonization_H(Mat &input_src, int *search_arr);
Mat ImgSkeletonization_V(Mat &input_src, int *search_arr);

#endif