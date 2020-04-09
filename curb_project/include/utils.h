#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <iostream>
#include <string>
#include <assert.h>
#include <boost/timer.hpp>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>

const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

void polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);

#endif