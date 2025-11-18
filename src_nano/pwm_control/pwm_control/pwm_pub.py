#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16

import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class CmdVelToPWMNode(Node):
    """
    ROS2: /cmd_vel → PCA9685(ESC, Servo)

    ESC (중립 1800 기준, +300μs 평행이동):
      - 전진: 1900~1950 μs
      - 후진: 1550~1600 μs                                                
      - neutral: 1800 μs

    Servo (측정 중립 1592 → 보드 중립 1800으로 보정):
      - 보간식: PWM = 11.6865 * θ + 1800
      - 범위: θ ∈ [-30, 30], PWM ∈ [1250, 1950]
    """

    def __init__(self):
        super().__init__('pwm_pub')

        # ===== 파라미터 =====
        self.servo_ch = self.declare_parameter("servo_channel", 15).value
        self.esc_ch   = self.declare_parameter("esc_channel", 14).value
        self.pwm_hz   = self.declare_parameter("pwm_hz", 50).value

        # ESC 설정
        self.esc_neutral = self.declare_parameter("esc_neutral_us", 1800).value
        self.PWM_FWD_MIN = self.declare_parameter("fwd_pwm_min", 1900).value
        # self.PWM_FWD_MAX = self.declare_parameter("fwd_pwm_max", 1950).value
        self.PWM_FWD_MAX = self.declare_parameter("fwd_pwm_max", 2250).value # 08172208 lsg 수정
        self.PWM_REV_MIN = self.declare_parameter("bwd_pwm_min", 1750).value #1550
        self.PWM_REV_MAX = self.declare_parameter("bwd_pwm_max", 1700).value #1600 

        # Servo 설정 (중립을 1800으로 보정)
        self.SERVO_DEG_MIN = self.declare_parameter("servo_deg_min", -30.0).value
        self.SERVO_DEG_MAX = self.declare_parameter("servo_deg_max", +30.0).value
        self.SERVO_PWM_MIN = self.declare_parameter("servo_pwm_min", 1458).value
        self.SERVO_PWM_MAX = self.declare_parameter("servo_pwm_max", 2158).value
        self.servo_m = self.declare_parameter("servo_m", 11.6865).value
        self.servo_c = self.declare_parameter("servo_c", 1800.0).value  # 중립 보정 = 1800

        # Ackermann
        self.wheelbase_m   = self.declare_parameter("wheelbase_m", 0.41).value
        self.v_epsilon_mps = self.declare_parameter("ack_min_speed", 0.01).value

        # ESC 회귀식 계수 (원래 1500 기준)
        self.k_fwd = 0.00858146
        self.b_fwd = -13.45125883
        self.k_rev = -0.0091066
        self.b_rev = 12.0119620
        self.ESC_SHIFT = 300  # 1500 → 1800

        # 안전/출력
        self.cmd_timeout = self.declare_parameter("cmd_timeout", 0.5).value
        self.print_pwm   = self.declare_parameter("print_pwm", True).value

        # ===== PCA9685 초기화 =====
        i2c = busio.I2C(SCL, SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = self.pwm_hz
        self.PERIOD_US = 1_000_000 // self.pwm_hz

        def set_us(ch: int, us: int):
            us = clamp(int(us), 0, self.PERIOD_US)
            duty = int(us * 0xFFFF / self.PERIOD_US)
            self.pca.channels[ch].duty_cycle = clamp(duty, 0, 0xFFFF)
        self.set_us = set_us

        # 상태
        self.tgt_esc_us   = self.esc_neutral
        self.tgt_servo_us = self._servo_deg_to_pwm(0.0)
        self.last_cmd_time = self.get_clock().now().nanoseconds / 1e9

        # 새로 추가: 계산된 조향각(라디안/도) 저장 변수
        self.current_delta_rad = 0.0
        self.current_delta_deg = 0.0

        # 퍼블리셔
        self.pub_esc   = self.create_publisher(UInt16, "/pwm/esc_us", 10)
        self.pub_servo = self.create_publisher(UInt16, "/pwm/servo_us", 10)

        # 구독
        self.create_subscription(Twist, "/cmd_vel", self.cb_cmd_vel, 10)

        # 타이머
        self.create_timer(1.0/50.0, self.on_timer)

        # 초기 중립 출력
        self._write_outputs(self.esc_neutral, self._servo_deg_to_pwm(0.0))
        self.get_logger().info("pwm_pub ready (ROS2). ESC/Servo 중립 모두 1800 μs 보정.")

    # ===== 매핑 =====
    def _servo_deg_to_pwm(self, deg: float) -> int:
        deg = clamp(deg, self.SERVO_DEG_MIN, self.SERVO_DEG_MAX)
        pwm = self.servo_m * deg + self.servo_c
        return int(clamp(pwm, self.SERVO_PWM_MIN, self.SERVO_PWM_MAX))

    def map_servo_from_ackermann(self, omega: float, v: float) -> int:
        # v가 너무 작으면 조향각 0 처리 (정지근처에서 각도 튀는 것 방지)
        if abs(v) < self.v_epsilon_mps:
            self.current_delta_rad = 0.0
            self.current_delta_deg = 0.0
            return self._servo_deg_to_pwm(0.0)

        # δ = atan(L * ω / v)
        self.current_delta_rad = math.atan(self.wheelbase_m * omega / v)
        self.current_delta_deg = -math.degrees(self.current_delta_rad) # 부호 반전
        return self._servo_deg_to_pwm(self.current_delta_deg)

    def map_esc_from_linear(self, v: float) -> int:
        if abs(v) < 0.05:
            return int(self.esc_neutral)
        if v > 0.0:
            pwm_orig = (v - self.b_fwd) / self.k_fwd
            pwm = pwm_orig + self.ESC_SHIFT
            pwm = clamp(pwm, self.PWM_FWD_MIN, self.PWM_FWD_MAX)
        else:
            pwm_orig = (self.b_rev - v) / (-self.k_rev)
            pwm = pwm_orig + self.ESC_SHIFT
            pwm = clamp(pwm, self.PWM_REV_MIN, self.PWM_REV_MAX)
        return int(pwm)

    # ===== 콜백/타이머 =====
    def cb_cmd_vel(self, msg: Twist):
        v = float(msg.linear.x)     # m/s
        w = float(msg.angular.z)    # rad/s
        self.tgt_esc_us   = self.map_esc_from_linear(v)
        self.tgt_servo_us = self.map_servo_from_ackermann(w, v)
        self.last_cmd_time = self.get_clock().now().nanoseconds / 1e9

    def on_timer(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_cmd_time) > self.cmd_timeout:
            esc = int(self.esc_neutral)
            srv = self._servo_deg_to_pwm(0.0)
            # 타임아웃일 때 각도도 0 유지
            self.current_delta_rad = 0.0
            self.current_delta_deg = 0.0
        else:
            esc = int(self.tgt_esc_us)
            srv = int(self.tgt_servo_us)
        self._write_outputs(esc, srv)

    # ===== 보드/토픽 출력 =====
    def _write_outputs(self, esc_us: int, servo_us: int):
        self.set_us(self.servo_ch, servo_us)
        self.set_us(self.esc_ch, esc_us)
        self.pub_esc.publish(UInt16(data=esc_us))
        self.pub_servo.publish(UInt16(data=servo_us))
        if self.print_pwm:
            self.get_logger().info(
                f"PWM → ESC:{esc_us:4d} μs | SERVO:{servo_us:4d} μs | δ={self.current_delta_deg:.2f} deg ({self.current_delta_rad:.3f} rad)"
            )

    def destroy_node(self):
        try:
            self._write_outputs(self.esc_neutral, self._servo_deg_to_pwm(0.0))
            time.sleep(1.0)
            self.pca.deinit()
        except Exception:
            pass
        super().destroy_node()
        self.get_logger().info("pwm_pub stopped (neutral & deinit).")

def main():
    rclpy.init()
    node = CmdVelToPWMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

#######

# import math
# import time
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from std_msgs.msg import UInt16

# import busio
# from board import SCL, SDA
# from adafruit_pca9685 import PCA9685


# def clamp(v, lo, hi):
#     return max(lo, min(hi, v))


# class CmdVelToPWMNode(Node):
#     """
#     ROS2: /cmd_vel → PCA9685(ESC, Servo)

#     설정(기본값):
#       - Servo 중립: 1500 μs
#       - ESC 중립: 1475 μs
#       - 오프셋(shift) 사용 안 함. 계산된 PWM 값을 그대로 사용.

#     ESC:
#       - |v| < 0.05 → 1475 μs (중립)
#       - v > 0 → 회귀식 기반 PWM → [fwd_pwm_min, fwd_pwm_max]로 클램프
#       - v < 0 → 회귀식 기반 PWM → [bwd_pwm_min, bwd_pwm_max]로 클램프

#     Servo:
#       - PWM = servo_m * θ + servo_c
#       - 기본: servo_c = 1500.0 → θ=0일 때 1500 μs
#       - θ 범위 및 PWM 범위는 파라미터로 클램프
#     """

#     def __init__(self):
#         super().__init__('pwm_pub')

#         # ===== 파라미터 =====
#         self.servo_ch = self.declare_parameter("servo_channel", 15).value
#         self.esc_ch   = self.declare_parameter("esc_channel", 14).value
#         self.pwm_hz   = self.declare_parameter("pwm_hz", 50).value

#         # ESC 설정 (중립 그대로 사용)
#         self.esc_neutral = self.declare_parameter("esc_neutral_us", 1430).value
#         self.PWM_FWD_MIN = self.declare_parameter("fwd_pwm_min", 1550).value
#         self.PWM_FWD_MAX = self.declare_parameter("fwd_pwm_max", 1950).value
#         self.PWM_REV_MIN = self.declare_parameter("bwd_pwm_min", 1100).value
#         self.PWM_REV_MAX = self.declare_parameter("bwd_pwm_max", 1450).value

#         # Servo 설정 (중립 1500)
#         self.SERVO_DEG_MIN = self.declare_parameter("servo_deg_min", -30.0).value
#         self.SERVO_DEG_MAX = self.declare_parameter("servo_deg_max", +30.0).value
#         # 필요시 실제 측정값으로 교체 가능
#         self.SERVO_PWM_MIN = self.declare_parameter("servo_pwm_min", 1100).value
#         self.SERVO_PWM_MAX = self.declare_parameter("servo_pwm_max", 1900).value
#         self.servo_m = self.declare_parameter("servo_m", 11.6865).value
#         self.servo_c = self.declare_parameter("servo_c", 1500.0).value  # 0deg → 1500 μs

#         # Ackermann
#         self.wheelbase_m   = self.declare_parameter("wheelbase_m", 0.41).value
#         self.v_epsilon_mps = self.declare_parameter("ack_min_speed", 0.01).value

#         # ESC 회귀식 계수 (v [m/s] → PWM [μs], 오프셋 없음 버전)
#         self.k_fwd = 0.00858146
#         self.b_fwd = -13.45125883
#         self.k_rev = -0.0091066
#         self.b_rev = 12.0119620

#         # 안전/출력
#         self.cmd_timeout = self.declare_parameter("cmd_timeout", 0.5).value
#         self.print_pwm   = self.declare_parameter("print_pwm", True).value

#         # ===== PCA9685 초기화 =====
#         i2c = busio.I2C(SCL, SDA)
#         self.pca = PCA9685(i2c)
#         self.pca.frequency = self.pwm_hz
#         self.PERIOD_US = 800_000 // self.pwm_hz

#         def set_us(ch: int, us: int):
#             us = clamp(int(us), 0, self.PERIOD_US)
#             duty = int(us * 0xFFFF / self.PERIOD_US)
#             self.pca.channels[ch].duty_cycle = clamp(duty, 0, 0xFFFF)

#         self.set_us = set_us

#         # 상태
#         self.tgt_esc_us   = self.esc_neutral
#         self.tgt_servo_us = self._servo_deg_to_pwm(0.0)
#         self.last_cmd_time = self.get_clock().now().nanoseconds / 1e9

#         self.current_delta_rad = 0.0
#         self.current_delta_deg = 0.0

#         # 퍼블리셔
#         self.pub_esc   = self.create_publisher(UInt16, "/pwm/esc_us", 10)
#         self.pub_servo = self.create_publisher(UInt16, "/pwm/servo_us", 10)

#         # 구독
#         self.create_subscription(Twist, "/cmd_vel", self.cb_cmd_vel, 10)

#         # 타이머
#         self.create_timer(1.0 / 50.0, self.on_timer)

#         # 초기 중립 출력
#         self._write_outputs(self.esc_neutral, self._servo_deg_to_pwm(0.0))
#         self.get_logger().info(
#             f"pwm_pub ready. ESC neutral={self.esc_neutral} μs, "
#             f"SERVO neutral={self.servo_c} μs (no offset)."
#         )

#     # ===== Servo 매핑 =====
#     def _servo_deg_to_pwm(self, deg: float) -> int:
#         deg = clamp(deg, self.SERVO_DEG_MIN, self.SERVO_DEG_MAX)
#         pwm = self.servo_m * deg + self.servo_c
#         return int(clamp(pwm, self.SERVO_PWM_MIN, self.SERVO_PWM_MAX))

#     def map_servo_from_ackermann(self, omega: float, v: float) -> int:
#         # 저속에서는 조향 0도로 고정
#         if abs(v) < self.v_epsilon_mps:
#             self.current_delta_rad = 0.0
#             self.current_delta_deg = 0.0
#             return self._servo_deg_to_pwm(0.0)

#         # δ = atan(L * ω / v)
#         self.current_delta_rad = math.atan(self.wheelbase_m * omega / v)
#         self.current_delta_deg = -math.degrees(self.current_delta_rad)  # 차량/서보 방향에 맞춰 필요 시 부호 조정
#         return self._servo_deg_to_pwm(self.current_delta_deg)

#     # ===== ESC 매핑 =====
#     def map_esc_from_linear(self, v: float) -> int:
#         # 정지 근처 → ESC 중립 PWM
#         if abs(v) < 0.05:
#             return int(self.esc_neutral)

#         if v > 0.0:
#             # 전진: 회귀식 결과를 바로 사용하고 FWD 범위로 클램프
#             pwm = (v - self.b_fwd) / self.k_fwd
#             pwm = clamp(pwm, self.PWM_FWD_MIN, self.PWM_FWD_MAX)
#         else:
#             # 후진: 회귀식 결과를 바로 사용하고 REV 범위로 클램프
#             pwm = (self.b_rev - v) / (-self.k_rev)
#             pwm = clamp(pwm, self.PWM_REV_MIN, self.PWM_REV_MAX)

#         return int(pwm)

#     # ===== 콜백/타이머 =====
#     def cb_cmd_vel(self, msg: Twist):
#         v = float(msg.linear.x)
#         w = float(msg.angular.z)

#         self.tgt_esc_us   = self.map_esc_from_linear(v)
#         self.tgt_servo_us = self.map_servo_from_ackermann(w, v)
#         self.last_cmd_time = self.get_clock().now().nanoseconds / 1e9

#     def on_timer(self):
#         now = self.get_clock().now().nanoseconds / 1e9
#         if (now - self.last_cmd_time) > self.cmd_timeout:
#             esc = int(self.esc_neutral)
#             srv = self._servo_deg_to_pwm(0.0)
#             self.current_delta_rad = 0.0
#             self.current_delta_deg = 0.0
#         else:
#             esc = int(self.tgt_esc_us)
#             srv = int(self.tgt_servo_us)

#         self._write_outputs(esc, srv)

#     # ===== 보드/토픽 출력 =====
#     def _write_outputs(self, esc_us: int, servo_us: int):
#         self.set_us(self.servo_ch, servo_us)
#         self.set_us(self.esc_ch, esc_us)

#         self.pub_esc.publish(UInt16(data=esc_us))
#         self.pub_servo.publish(UInt16(data=servo_us))

#         if self.print_pwm:
#             self.get_logger().info(
#                 f"PWM → ESC:{esc_us:4d} μs | SERVO:{servo_us:4d} μs "
#                 f"| δ={self.current_delta_deg:.2f} deg ({self.current_delta_rad:.3f} rad)"
#             )

#     def destroy_node(self):
#         try:
#             self._write_outputs(self.esc_neutral, self._servo_deg_to_pwm(0.0))
#             time.sleep(1.0)
#             self.pca.deinit()
#         except Exception:
#             pass

#         super().destroy_node()
#         self.get_logger().info("pwm_pub stopped (neutral & deinit).")


# def main():
#     rclpy.init()
#     node = CmdVelToPWMNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
