#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class ImageConverter(Node):
    def __init__(self):
        super().__init__('image_converter')
        
        # Create publisher for compressed image
        self.camera_down = self.create_publisher(
            CompressedImage, 
            '/downward_camera/image/compressed', 
            10
        )
        
        #work in progresss
        
        
        # Initialize video capture from default camera (0)
        self.cap0 = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(2)
        
        
        if not self.cap0.isOpened():
            self.get_logger().error('Failed to open camera 0!')
            return
        
        if not self.cap1.isOpened():
            self.get_logger().error('Failed to open camera 1!')
            return
        
        # Set camera properties (optional - adjust as needed)
        self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap0.set(cv2.CAP_PROP_FPS, 30)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap1.set(cv2.CAP_PROP_FPS, 30)
        
        # Create timer to publish images at regular intervals
        timer_period = 0.033  # 30 Hz (seconds)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info('Image Converter Node started - Publishing compressed images')
    
    def timer_callback(self):
        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        
        if not ret0 or not ret1:
            self.get_logger().warn('Failed to capture frame from one or both cameras')
            return
        if not ret0:
            self.get_logger().warn('Failed to capture frame from camera 0')
            return
        if not ret1:
            self.get_logger().warn('Failed to capture frame from camera 1')
            return
        
        # Create CompressedImage message for camera 0
        msg0 = CompressedImage()
        msg0.header.stamp = self.get_clock().now().to_msg()
        msg0.header.frame_id = 'camera_frame_0'
        msg0.format = 'jpeg'
        
        # Create CompressedImage message for camera 1
        msg1 = CompressedImage()
        msg1.header.stamp = self.get_clock().now().to_msg()
        msg1.header.frame_id = 'camera_frame_1'
        msg1.format = 'jpeg'
        
        # Encode frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result0, encoded_image0 = cv2.imencode('.jpg', frame0, encode_param)
        result1, encoded_image1 = cv2.imencode('.jpg', frame1, encode_param)
        
        if result0:
            msg0.data = np.array(encoded_image0).tobytes()
            self.publisher.publish(msg0)
        else:
            self.get_logger().warn('Failed to encode image from camera 0')
        
        if result1:
            msg1.data = np.array(encoded_image1).tobytes()
            self.publisher.publish(msg1)
        else:
            self.get_logger().warn('Failed to encode image from camera 1')
    
    def destroy_node(self):
        # Release the camera when shutting down
        if hasattr(self, 'cap0'):
            self.cap0.release()
        if hasattr(self, 'cap1'):
            self.cap1.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
