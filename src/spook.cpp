#include "spook.hpp"
#include <librealsense2/rs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <chrono>

DeepSpook::DeepSpook() {
  rs2::config cfg;

  cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
  // cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

  profile_ = pipeline_.start(cfg);
  float depth_scale_ = getDepthScale(profile_.get_device());
  std::cout << depth_scale_ << std::endl;
}

float DeepSpook::getDepthScale(rs2::device dev) {
  // Go over the device's sensors
  for (rs2::sensor& sensor : dev.query_sensors()) {
    // Check if the sensor if a depth sensor
    if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
    {
      return dpt.get_depth_scale();
    }
  }
  throw std::runtime_error("Device does not have a depth sensor");
}

bool DeepSpook::profileChanged(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev) {
  for (auto&& sp : prev) {
    auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
    if (itr == std::end(current)) {
      return true;
    }
  }
  return false;
}

void DeepSpook::addGhosts(cv::Mat rgb_frame, cv::Mat depth_mask, cv::Mat ghost_frame) {
  cv::add(rgb_frame, ghost_frame, rgb_frame, depth_mask);
}

int DeepSpook::run() try {
  rs2_stream align_to = RS2_STREAM_COLOR;
  rs2::align align(align_to);

  cv::VideoCapture vid = cv::VideoCapture("../vids/ghosts_fullres.mp4");

  auto start_time_ = std::chrono::system_clock::now();

  while (true) {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = now-start_time_;
    
    rs2::frameset frameset = pipeline_.wait_for_frames();

    if (profileChanged(pipeline_.get_active_profile().get_streams(), profile_.get_streams())) {
        //If the profile was changed, update the align object, and also get the new device's depth scale
        profile_ = pipeline_.get_active_profile();
        rs2_stream align_to = RS2_STREAM_COLOR;
        align = rs2::align(align_to);
        depth_scale_ = getDepthScale(profile_.get_device());
    }

    //Get processed aligned frame
    auto processed = align.process(frameset);

    // Trying to get both other and aligned depth frames
    rs2::video_frame rgb_frame = processed.first(align_to);
    rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

    //If one of them is unavailable, continue iteration
    if (!aligned_depth_frame || !rgb_frame) {
      continue;
    }

    cv::Mat rgb_cv_frame(cv::Size(1280, 720), CV_8UC3, (void*)rgb_frame.get_data(), cv::Mat::AUTO_STEP);
    cv::Mat depth_cv_frame(cv::Size(1280, 720), CV_16U, (void*)aligned_depth_frame.get_data(), cv::Mat::AUTO_STEP);
    // depth_cv_frame.convertTo(depth_cv_frame, CV_8UC1, 255.0/1000); // Uncomment to imshow depth frame

    cv::Mat thresh;
    cv::inRange(depth_cv_frame, cv::Scalar(1.0), cv::Scalar(3000.0), thresh);
    cv::bitwise_not(thresh, thresh);

    if (elapsed_seconds.count() > 5.0) {
      cv::Mat vid_frame;
      vid.read(vid_frame);
      vid.read(vid_frame);
      cv::resize(vid_frame, vid_frame, cv::Size(1280, 720));

      addGhosts(rgb_cv_frame, thresh, vid_frame);
    }

    cv::namedWindow("DeepSpook", cv::WINDOW_AUTOSIZE);
    cv::imshow("DeepSpook", rgb_cv_frame);

    cv::waitKey(30);
  }
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

int main(int argc, char * argv[])
{
  DeepSpook* way2spooky4me = new DeepSpook();
  int ret = way2spooky4me->run();

  return ret;
}
