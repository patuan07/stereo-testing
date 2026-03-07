#ifndef STEREO_IMAGE_PROC__RECT_PREPROCESS_NODE_HPP_
#define STEREO_IMAGE_PROC__RECT_PREPROCESS_NODE_HPP_

#include <memory>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace stereo_image_proc
{

class RectPreprocessNode : public rclcpp::Node
{
public:
  explicit RectPreprocessNode(const rclcpp::NodeOptions & options);

private:
  // Subscribers
  image_transport::Subscriber sub_left_;
  image_transport::Subscriber sub_right_;

  // Publishers
  image_transport::Publisher pub_left_;
  image_transport::Publisher pub_right_;

  // Preprocessing helpers
  cv::Mat histStretch(const cv::Mat & src);
  cv::Mat applyClahe(const cv::Mat & src, double clip_limit, int tile_size);
  cv::Mat preprocess(const cv::Mat & src);

  // Callbacks
  void leftCb(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void rightCb(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void processAndPublish(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    image_transport::Publisher & publisher);
};

}  // namespace stereo_image_proc

#endif  // STEREO_IMAGE_PROC__RECT_PREPROCESS_NODE_HPP_
