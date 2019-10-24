#ifndef PLUGIN_UNDISTORT_H
#define PLUGIN_UNDISTORT_H

#include <Eigen/Dense>
#include <camera_calibration.h>
#include <image.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <visionplugin.h>

class PluginUndistort : public VisionPlugin {
protected:
  CameraParameters &camera_params;

  std::unique_ptr<VarList> settings;
  std::unique_ptr<VarBool> enable;
  std::unique_ptr<VarStringEnum> interpolation_mode;

  //! Maps from distorted to undistorted image
  cv::Mat undistort_map_x;
  cv::Mat undistort_map_y;

public:
  PluginUndistort(FrameBuffer *buffer, CameraParameters &camera_params);
  ~PluginUndistort() = default;

  ProcessResult process(FrameData *data, RenderOptions *options) override;
  VarList *getSettings() override;
  std::string getName() override;

private:
  // TODO: maybe should be done in the camera calibration?
  //
  // TODO: will need to call this whenever camera parameters have
  // changed
  void calculateUndistortedImageBounds(const int width, const int height,
                                       Eigen::Vector2d &upper_left,
                                       Eigen::Vector2d &lower_right);

  void makeUndistortMap(const int undistorted_width, const int undistorted_height);
};

#endif /* PLUGIN_UNDISTORT_H */
