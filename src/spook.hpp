// DeepSpook
// Copyright 2019 Lucas Watson (@lkwatson)
//
// Portions of code copied and modified from librealsense align-advanced example.
// See https://github.com/IntelRealSense/librealsense
// License: Apache 2.0
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>
#include "opencv2/imgproc/imgproc.hpp"

class DeepSpook {
public:
  DeepSpook();
  int run();

private:
  float getDepthScale(rs2::device);
  bool profileChanged(const std::vector<rs2::stream_profile>&, const std::vector<rs2::stream_profile>&);
  void addGhosts(cv::Mat, cv::Mat, cv::Mat);

private:
  rs2::pipeline pipeline_;
  rs2::pipeline_profile profile_;

  float foreground_split_ = 3.0f;
  float depth_scale_;
};