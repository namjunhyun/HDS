"""
compressed 이미지를 raw로 변환해서 /local/ 네임스페이스로 relay
conda deactivate 후 시스템 Python으로 실행
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5
)

class CompressedRelay(Node):
    def __init__(self, in_topic, out_topic, encoding='bgr8'):
        name = 'relay_' + out_topic.strip('/').replace('/', '_')
        super().__init__(name)
        self.bridge   = CvBridge()
        self.encoding = encoding
        self.pub = self.create_publisher(Image, out_topic, QOS)
        self.sub = self.create_subscription(CompressedImage, in_topic, self.cb, QOS)
        self.get_logger().info(f'{in_topic} → {out_topic}')

    def cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        out = self.bridge.cv2_to_imgmsg(img, self.encoding)
        out.header = msg.header
        self.pub.publish(out)

if __name__ == '__main__':
    rclpy.init()
    color = CompressedRelay(
        '/camera/color/image_raw/compressed',
        '/local/color/image_raw',
        encoding='bgr8'
    )
    depth = CompressedRelay(
        '/camera/depth/image_rect_raw/compressed',
        '/local/depth/image_rect_raw',
        encoding='passthrough'
    )
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(color)
    executor.add_node(depth)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
