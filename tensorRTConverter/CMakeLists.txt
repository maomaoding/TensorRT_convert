cmake_minimum_required(VERSION 2.8.3)
project(convert_model)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(CUDA)
include_directories(
  ${CUDA_INCLUDE_DIRS}
 ./ 
 layers/
 /usr/local/TensorRT/include
 #/datastore/data/zhq/software/TensorRT-5.1.5.0/include
 /usr/local/cuda/include)
link_directories(
/usr/lib/aarch64-linux-gnu 
/usr/local/TensorRT/lib
#/datastore/data/zhq/software/TensorRT-5.1.5.0/lib
/usr/local/cuda/lib64 )


file(GLOB
     cudaSrc
     *.cu
     layers/*.cu
     layers/*.cpp
     utils.cpp
    )
cuda_add_library(util_${PROJECT_NAME} SHARED ${cudaSrc})
target_link_libraries(util_${PROJECT_NAME}
                      ${CUDA_LIBRARIES}
                      nvcaffe_parser
                      nvonnxparser
                      nvinfer
                      nvinfer_plugin
                      nvparsers)

add_executable(${PROJECT_NAME}
buildTensorRTModel.cpp
pluginFactory.cpp
tensorNet.cpp
)
target_link_libraries(${PROJECT_NAME} util_${PROJECT_NAME})
