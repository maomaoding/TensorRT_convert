#include "imgskeletonization.h"

Mat ImgSkeletonization(Mat &input_src,Mat & output_dst, int number)
{
	output_dst = input_src.clone();
	int search_array[]= { 0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
		1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
		0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
		1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
		1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
		1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
		0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
		1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
		0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
		1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
		1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
		1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
		1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
		1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0 };
	for (size_t i = 0; i < number; i++)
	{
		ImgSkeletonization_H(output_dst, &search_array[0]);
		ImgSkeletonization_V(output_dst, &search_array[0]);
		
	}
	return output_dst;
}

Mat ImgSkeletonization_H(Mat &input_src, int *search_arr)
{
	int h = input_src.rows;
	int w = input_src.cols;
	bool NEXT = true;
	for (size_t j = 1; j < w - 1; j++)//注意边界问题！！！！！！
	{
		for (size_t i = 1; i < h - 1; i++)
		{
			if (!NEXT)
				NEXT = true;
			else
			{
				int judge_value;
				if (1 <i < h - 1)
					judge_value = input_src.at<uchar>(i - 1, j) + input_src.at<uchar>(i, j) + input_src.at<uchar>(i + 1, j);
				else
					judge_value = 1;
				if (input_src.at<uchar>(i, j) == 0 && judge_value != 0)
				{
					int a[9] = { 1,1,1,1,1,1,1,1,1};
					for (size_t m = 0; m < 3; m++)
					{
						for (size_t n = 0; n < 3; n++)
						{
							if ((0 <= (i - 1 + m) < h) && (0 <= (j - 1 + n) < w) && input_src.at<uchar>(i - 1 + m, j - 1 + n) == 0)
								a[m * 3 + n] = 0;
						}
					}
					int sum_value = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128;
					input_src.at<uchar>(i, j) = search_arr[sum_value] * 255;
					if (search_arr[sum_value] == 1)
						NEXT = false;
				}
			}
		}
	}
	return input_src;
}
Mat ImgSkeletonization_V(Mat &input_src, int *search_arr)
{
	int h = input_src.rows;
	int w = input_src.cols;
	bool NEXT = true;
	for (size_t i = 1; i < h - 1; i++)//注意边界问题！！！！！！
	{
		for (size_t j = 1; j < w - 1; j++)
		{
			if (!NEXT)
				NEXT = true;
			else
			{
				int judge_value;
				if (1 < j <w - 1)
					judge_value = input_src.at<uchar>(i, j - 1) + input_src.at<uchar>(i, j) + input_src.at<uchar>(i, j + 1);
				else
					judge_value = 1;
				if (input_src.at<uchar>(i, j) == 0 && judge_value != 0)
				{
					int a[9] = {1,1,1,1,1,1,1,1,1 };
					for (size_t m = 0; m < 3; m++)
					{
						for (size_t n = 0; n < 3; n++)
						{
							if ((0 <= (i - 1 + m) < h) && (0 <= (j - 1 + n) < w) && input_src.at<uchar>(i - 1 + m, j - 1 + n) == 0)
								a[m * 3 + n] = 0;
						}
					}
					int sum_value = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128;
					input_src.at<uchar>(i, j) = search_arr[sum_value] * 255;
					if (search_arr[sum_value] == 1)
						NEXT = false;
				}
			}
		}
	}
	return input_src;
}