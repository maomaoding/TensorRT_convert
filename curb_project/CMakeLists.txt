cmake_minimum_required(VERSION 2.8)
project(curb_perception)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp image_transport sensor_msgs cv_bridge std_msgs tf camera_calibration_parsers
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lanenet
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(include ${catkin_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
	if(it MATCHES "vtk*")
	list(APPEND vtk_flags ${it})
	endif()
endforeach()
foreach(d ${vtk_flags})
	remove_definitions(-D${d})
endforeach()


find_package(CUDA  REQUIRED)
find_package(OpenCV REQUIRED)

option(denseaspp "use denseaspp model" off)
option(scnncaffe_udlr "use scnncaffe_udlr model" off)
option(vggscnn "use vggscnn model" on)

if(denseaspp)
	add_definitions(-Ddenseaspp)
endif(denseaspp)

if(scnncaffe_udlr)
	add_definitions(-Dscnncaffe_udlr)
endif(scnncaffe_udlr)

if(vggscnn)
	add_definitions(-Dvggscnn)
endif(vggscnn)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(/usr/include/eigen3)

LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB DIR_SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
					${CMAKE_CURRENT_LIST_DIR}/src/utils/*.cpp)

cuda_add_executable(curb_perception main.cpp ${DIR_SRCS})


target_link_libraries(curb_perception libnvcaffe_parser.so)
target_link_libraries(curb_perception libnvinfer.so)
target_link_libraries(curb_perception libnvinfer_plugin.so)
target_link_libraries(curb_perception libnvparsers.so)

target_link_libraries(curb_perception ${OpenCV_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES})