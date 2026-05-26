"""
HDS ROS2 Bridge (시스템 Python 3.12으로 실행)
/camera/imu + /local/color/image_raw → /dev/shm에 저장
"""
import sys, threading, collections
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge

IMU_WINDOW = 30

class BridgeNode(Node):
    def __init__(self):
        super().__init__('hds_bridge')
        self.bridge      = CvBridge()
        self.imu_buf     = collections.deque(maxlen=IMU_WINDOW)
        self._last_accel = np.zeros(3)
        self._lock       = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(Imu, '/camera/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Image, '/local/color/image_raw', self._color_cb, sensor_qos)
        self.get_logger().info("Bridge 시작: /camera/imu + /local/color/image_raw → /dev/shm")

    def _imu_cb(self, msg: Imu):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        with self._lock:
            self._last_accel = np.array([ax, ay, az])
            self.imu_buf.append(np.array([gx, gy, gz, ax, ay, az]))
            if len(self.imu_buf) == IMU_WINDOW:
                np.save('/dev/shm/imu_buffer.npy', np.array(self.imu_buf))

    def _color_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite('/dev/shm/latest_frame.jpg', bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")

if __name__ == '__main__':
    rclpy.init()
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
