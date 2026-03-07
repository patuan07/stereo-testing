#include <memory>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/imgproc.hpp"

#include <stereo_image_proc/rect_preprocess_node.hpp>

#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace stereo_image_proc
{

RectPreprocessNode::RectPreprocessNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("rect_preprocess_node", options)
{
  // Declare preprocessing parameters - all default true
  this->declare_parameter("preprocess.clahe",            true);
  this->declare_parameter("preprocess.hist_stretch",     true);
  this->declare_parameter("preprocess.clahe_clip_limit", 2.0);
  this->declare_parameter("preprocess.clahe_tile_size",  8);

  // Publishers for preprocessed MONO8 images
  pub_left_  = image_transport::create_publisher(this, "left/image_rect_preprocessed");
  pub_right_ = image_transport::create_publisher(this, "right/image_rect_preprocessed");

  // Subscribers to rectified mono images from stereo_image_proc
  sub_left_ = image_transport::create_subscription(
    this, "left/image_rect",
    std::bind(&RectPreprocessNode::leftCb, this, std::placeholders::_1),
    "raw");

  sub_right_ = image_transport::create_subscription(
    this, "right/image_rect",
    std::bind(&RectPreprocessNode::rightCb, this, std::placeholders::_1),
    "raw");
}

// ---------------------------------------------------------------------------
// Preprocessing helpers
// ---------------------------------------------------------------------------

cv::Mat RectPreprocessNode::histStretch(const cv::Mat & src)
{
  // Stretch single-channel MONO8 image to full [0, 255] range.
  // Compensates for low-contrast underwater imagery where the useful
  // signal is compressed into a narrow intensity range.
  double lo, hi;
  cv::minMaxLoc(src, &lo, &hi);

  cv::Mat dst;
  if (hi > lo) {
    src.convertTo(dst, CV_8U, 255.0 / (hi - lo), -lo * 255.0 / (hi - lo));
  } else {
    dst = src.clone();
  }
  return dst;
}

cv::Mat RectPreprocessNode::applyClahe(const cv::Mat & src, double clip_limit, int tile_size)
{
  // Apply CLAHE directly on the single-channel greyscale image.
  // No LAB conversion needed since the input is already MONO8.
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
    clip_limit,
    cv::Size(tile_size, tile_size));

  cv::Mat dst;
  clahe->apply(src, dst);
  return dst;
}

cv::Mat RectPreprocessNode::preprocess(const cv::Mat & src)
{
  const bool do_hist_stretch = this->get_parameter("preprocess.hist_stretch").as_bool();
  const bool do_clahe        = this->get_parameter("preprocess.clahe").as_bool();
  const double clip_limit    = this->get_parameter("preprocess.clahe_clip_limit").as_double();
  const int tile_size        = this->get_parameter("preprocess.clahe_tile_size").as_int();

  cv::Mat img = src.clone();

  if (do_hist_stretch) {
    img = histStretch(img);
  }

  if (do_clahe) {
    img = applyClahe(img, clip_limit, tile_size);
  }

  return img;
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void RectPreprocessNode::processAndPublish(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  image_transport::Publisher & publisher)
{
  // Skip processing if nobody is listening
  if (publisher.getNumSubscribers() == 0) {
    return;
  }

  // Decode as MONO8 - image_rect from stereo_image_proc is already greyscale
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // Apply preprocessing
  cv::Mat processed = preprocess(cv_ptr->image);

  // Publish - preserve original header so timestamp and frame_id are unchanged.
  // This is critical for the disparity node's message_filters synchronizer to
  // correctly match left/right images by timestamp.
  cv_bridge::CvImage out_msg;
  out_msg.header   = msg->header;
  out_msg.encoding = sensor_msgs::image_encodings::MONO8;
  out_msg.image    = processed;

  publisher.publish(out_msg.toImageMsg());
}

void RectPreprocessNode::leftCb(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  processAndPublish(msg, pub_left_);
}

void RectPreprocessNode::rightCb(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  processAndPublish(msg, pub_right_);
}

}  // namespace stereo_image_proc

// Register as a composable component so it can be loaded into the
// stereo_image_proc_container alongside DisparityNode and PointCloudNode,
// enabling zero-copy intra-process image passing.
RCLCPP_COMPONENTS_REGISTER_NODE(stereo_image_proc::RectPreprocessNode)
