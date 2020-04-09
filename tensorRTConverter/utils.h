/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#ifndef __UTILS_H__
#define __UTILS_H__

#include <cuda_runtime.h>
#include <memory>
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <algorithm>
#include<vector>
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#define DIV_THEN_CEIL(x, y) (((x) + (y)-1) / (y))
void getFiles(const char *path, std::vector<std::string> &files);
std::vector<std::string> supersplit(const std::string& s, const std::string& c);
std::string getModelDir(std::string input);

inline float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

//————————————————————————————————————————————————————————————————————————————————————————
//proposal layer fucs
void nms_cpu(const int num_boxes,
             const float boxes[],
             int index_out[],
             int *const num_out,
             const int base_index,
             const float nms_thresh,
             const int max_num_out);

void nms_gpu(const int num_boxes,
             const float boxes_gpu[],
             int *const p_mask,
             int index_out_cpu[],
             int *const num_out,
             const int base_index,
             const float nms_thresh,
             const int max_num_out);
//————————————————————————————————————————————————————————————————————————————————————————
// CUDA: use 512 threads per block
const int TENSORRT_CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int TENSORRT_GET_BLOCKS(const int N) {
  return (N + TENSORRT_CUDA_NUM_THREADS - 1) / TENSORRT_CUDA_NUM_THREADS;
}
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
#endif // MODULES_PERCEPTION_OBSTACLE_CAMERA_DETECTOR_COMMON_UTIL_H_
