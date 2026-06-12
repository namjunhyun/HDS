"""
ROS2 완화 노드 — /dev/shm/hds_score 를 읽어 완화 동작 발행. 정책은 mitigation_policy.py.

구조 이유: hds_g1_local.py는 conda Python 3.9라 rclpy 발행 불가 → /dev/shm 파일로 점수 전달
(hds_ros_bridge와 동일 패턴). 이 노드(시스템 Python 3.12, ROS2)가 그 파일을 타이머로 읽음.

입력 : /dev/shm/hds_score   ("<hds_score> <alert> <timestamp>")  — hds_g1_local.py가 기록
출력 : /hds/score (Float32, 리퍼블리시)·/hds/speed_scale (Float32)·/hds/trigger_reloc (Bool)
연결(TODO): G1 보행 컨트롤러가 /hds/speed_scale 를 속도 게인으로 반영. reloc은 ORB-SLAM3 서비스로.
로그 : (t, hds, speed_scale, reloc, active) CSV → traj_eval 연계 분석.

실행: ros2 run ... 또는  python3 ros2_mitigation_node.py   (ROS2 환경에서만)
"""
import os, csv, time

SCORE_SHM   = "/dev/shm/hds_score"
READ_HZ     = 20.0
STALE_SEC   = 1.0          # 점수 갱신이 이보다 오래 끊기면 완화 해제(안전)

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Bool
    _ROS = True
except Exception:
    _ROS = False

from mitigation_policy import MitigationPolicy, MitigationConfig


def read_score(path=SCORE_SHM):
    """/dev/shm 점수 파싱 → (hds, alert, ts) 또는 None."""
    try:
        v = open(path).read().split()
        return float(v[0]), int(v[1]), float(v[2])
    except Exception:
        return None


if _ROS:
    class MitigationNode(Node):
        def __init__(self):
            super().__init__("hds_mitigation")
            self.policy = MitigationPolicy(MitigationConfig())
            self.score_pub = self.create_publisher(Float32, "/hds/score", 10)
            self.scale_pub = self.create_publisher(Float32, "/hds/speed_scale", 10)
            self.reloc_pub = self.create_publisher(Bool, "/hds/trigger_reloc", 10)
            self._log = csv.writer(open(f"mitig_log_{time.strftime('%Y%m%d_%H%M%S')}.csv", "w", newline=""))
            self._log.writerow(["t", "hds", "speed_scale", "reloc", "active"])
            self.create_timer(1.0 / READ_HZ, self._tick)
            self.get_logger().info(f"HDS 완화 노드 시작: {SCORE_SHM} 폴링 @ {READ_HZ:.0f}Hz")

        def _tick(self):
            r = read_score()
            now = time.time()
            if r is None or (now - r[2]) > STALE_SEC:
                # 점수 없음/오래됨 → 안전하게 완화 해제(정속)
                self.scale_pub.publish(Float32(data=1.0))
                return
            hds, _alert, _ts = r
            a = self.policy.decide(hds)
            self.score_pub.publish(Float32(data=float(hds)))
            self.scale_pub.publish(Float32(data=float(a.speed_scale)))
            if a.trigger_reloc:
                self.reloc_pub.publish(Bool(data=True))
                self.get_logger().warn(f"relocalization 트리거 (hds={hds:.2f})")
            self._log.writerow([f"{now:.3f}", f"{hds:.3f}", f"{a.speed_scale:.3f}",
                                int(a.trigger_reloc), int(a.active)])

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
        print("rclpy 없음 — ROS2 환경에서 실행. read_score()는 단독 테스트 가능:")
        print("  /dev/shm/hds_score 예시 기록 후 read_score() 확인")
