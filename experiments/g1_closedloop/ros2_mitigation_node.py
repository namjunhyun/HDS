"""
ROS2 완화 노드 (스켈레톤) — 로봇/ROS 환경에서만 실행. 정책 로직은 mitigation_policy.py.

⚠ 이 파일은 G1 + ROS2 환경에서 토픽을 연결해야 동작하는 골격이다(여기선 미실행).
연결 지점(TODO):
  · 입력: HDS 점수 토픽 /hds/score (std_msgs/Float32) — hds_g1_local.py에 퍼블리셔 추가 필요.
  · 출력: 속도 스케일을 G1 보행 컨트롤러에 전달 (cmd_vel 게인 또는 전용 토픽).
  · relocalization 트리거: ORB-SLAM3 서비스/토픽으로 호출.

로그: 매 콜백마다 (t, hds, speed_scale, reloc) CSV → 조건 C1/C2 비교 및 traj_eval 연계.
"""
import csv, time

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Bool
    from geometry_msgs.msg import Twist
    _ROS = True
except Exception:
    _ROS = False

from mitigation_policy import MitigationPolicy, MitigationConfig


if _ROS:
    class MitigationNode(Node):
        def __init__(self):
            super().__init__("hds_mitigation")
            self.policy = MitigationPolicy(MitigationConfig())
            self._last_cmd = Twist()
            self._log = csv.writer(open(f"mitig_log_{time.strftime('%Y%m%d_%H%M%S')}.csv", "w", newline=""))
            self._log.writerow(["t", "hds", "speed_scale", "reloc", "active"])

            self.create_subscription(Float32, "/hds/score", self._on_score, 10)
            # TODO: 베이스 컨트롤러가 구독하는 cmd_vel을 받아 스케일해서 재퍼블리시하거나,
            #       전용 속도게인 토픽으로 보낼 것.
            self.scale_pub = self.create_publisher(Float32, "/hds/speed_scale", 10)
            self.reloc_pub = self.create_publisher(Bool, "/hds/trigger_reloc", 10)
            self.get_logger().info("HDS 완화 노드 시작: /hds/score 구독")

        def _on_score(self, msg: Float32):
            a = self.policy.decide(float(msg.data))
            self.scale_pub.publish(Float32(data=float(a.speed_scale)))
            if a.trigger_reloc:
                self.reloc_pub.publish(Bool(data=True))
                self.get_logger().warn("relocalization 트리거")
            self._log.writerow([f"{time.time():.3f}", f"{msg.data:.3f}",
                                f"{a.speed_scale:.3f}", int(a.trigger_reloc), int(a.active)])

    def main():
        rclpy.init(); node = MitigationNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node(); rclpy.shutdown()

    if __name__ == "__main__":
        main()
else:
    if __name__ == "__main__":
        print("rclpy 없음 — 이 스켈레톤은 ROS2 환경에서 실행. 정책 단위테스트: "
              "python3 mitigation_policy.py --selftest")
