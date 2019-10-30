# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
cmake_minimum_required(VERSION 3.1.0)

project(deepspook)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME} src/spook.cpp)
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} ${OpenCV_LIBS})
