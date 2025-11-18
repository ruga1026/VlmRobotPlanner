#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RampCmdVelPub(Node):
    def __init__(self):
        super().__init__('cmdvel')

        # ===== 선형 속도 파라미터 =====
        self.declare_parameter('rate_hz', 5.0)            # 발행 주기 [Hz]
        self.declare_parameter('fwd_max', 1.8)            # 전진 최대속도 [m/s]
        self.declare_parameter('bwd_max', 1.0)            # 후진 최대속도(양수) [m/s]
        self.declare_parameter('accel', 0.3)              # 선형 가감속 [m/s^2]
        self.declare_parameter('hold_time', 0.0)          # 선형 목표속도에서 유지 [s]

        # ===== 조향(각속도) 파라미터 =====
        self.declare_parameter('ang_max', 0.8)            # 조향 각속도 최대치 [rad/s]
        self.declare_parameter('ang_accel', 0.4)          # 각속도 가감속 [rad/s^2]
        self.declare_parameter('ang_hold_time', 0.0)      # 각속도 목표에서 유지 [s]
        self.declare_parameter('ang_bias', 0.0)           # 각속도 중심 오프셋 [rad/s] (기본 0)

        # ===== 파라미터 읽기 =====
        self.rate_hz   = float(self.get_parameter('rate_hz').value)
        self.fwd_max   = float(self.get_parameter('fwd_max').value)
        self.bwd_max   = float(self.get_parameter('bwd_max').value)
        self.accel     = float(self.get_parameter('accel').value)
        self.hold_time = float(self.get_parameter('hold_time').value)

        self.ang_max       = float(self.get_parameter('ang_max').value)
        self.ang_accel     = float(self.get_parameter('ang_accel').value)
        self.ang_hold_time = float(self.get_parameter('ang_hold_time').value)
        self.ang_bias      = float(self.get_parameter('ang_bias').value)

        # 퍼블리셔
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ===== 선형 상태 =====
        self.dt = 1.0 / self.rate_hz
        self.v  = 0.0           # 현재 선형 속도
        self.v_target = 0.0     # 선형 목표
        self.v_phase = 0        # 0:↑(→+fwd), 1:↓(→0), 2:↑(→-bwd), 3:↓(→0)
        self.v_hold_timer = 0.0

        # ===== 조향(각속도) 상태 =====
        self.wz = 0.0           # 현재 각속도
        self.wz_target = 0.0    # 각속도 목표
        self.wz_phase = 0       # 0:↑(→+ang), 1:↓(→0), 2:↑(→-ang), 3:↓(→0)
        self.wz_hold_timer = 0.0

        self.get_logger().info(
            f'램프 cmd_vel 시작 | rate={self.rate_hz:.1f} Hz\n'
            f'  [Linear] fwd_max={self.fwd_max:.2f} m/s, bwd_max={self.bwd_max:.2f} m/s, accel={self.accel:.2f} m/s^2, hold={self.hold_time:.2f}s\n'
            f'  [Angular] ang_max={self.ang_max:.2f} rad/s, ang_accel={self.ang_accel:.2f} rad/s^2, hold={self.ang_hold_time:.2f}s, bias={self.ang_bias:.2f}'
        )

        self.timer = self.create_timer(self.dt, self.on_timer)

    def _ramp_to(self, curr: float, target: float, step: float) -> float:
        """step만큼 등가속으로 target을 향해 이동"""
        if abs(target - curr) <= step:
            return target
        return curr + (step if target > curr else -step)

    def _advance_phase_if_hold_done(self, reached: bool, hold_time: float, hold_timer_attr: str, phase_attr: str):
        """목표 도달 시 hold_time 유지 후 다음 phase로"""
        if not reached:
            return
        t = getattr(self, hold_timer_attr)
        if hold_time > 0.0:
            t += self.dt
            if t + 1e-9 >= hold_time:
                t = 0.0
                setattr(self, phase_attr, (getattr(self, phase_attr) + 1) % 4)
            setattr(self, hold_timer_attr, t)
        else:
            setattr(self, phase_attr, (getattr(self, phase_attr) + 1) % 4)

    def on_timer(self):
        # ===== 선형 목표 설정 =====
        if self.v_phase == 0:      # 0 -> +fwd_max
            self.v_target = self.fwd_max
        elif self.v_phase == 1:    # +fwd_max -> 0
            self.v_target = 0.0
        elif self.v_phase == 2:    # 0 -> -bwd_max
            self.v_target = -self.bwd_max
        else:                      # -bwd_max -> 0
            self.v_target = 0.0

        # ===== 각속도 목표 설정(선형과 독립) =====
        if self.wz_phase == 0:     # 0 -> +ang_max
            self.wz_target = +self.ang_max
        elif self.wz_phase == 1:   # +ang_max -> 0
            self.wz_target = 0.0
        elif self.wz_phase == 2:   # 0 -> -ang_max
            self.wz_target = -self.ang_max
        else:                      # -ang_max -> 0
            self.wz_target = 0.0

        # ===== 램프 갱신 =====
        v_step  = self.accel * self.dt
        wz_step = self.ang_accel * self.dt

        v_prev  = self.v
        wz_prev = self.wz

        self.v  = self._ramp_to(self.v,  self.v_target,  v_step)
        self.wz = self._ramp_to(self.wz, self.wz_target, wz_step)

        # 목표 도달 시 phase 전환(각각 독립)
        self._advance_phase_if_hold_done(abs(self.v - self.v_target)  < 1e-6,
                                         self.hold_time, 'v_hold_timer', 'v_phase')
        self._advance_phase_if_hold_done(abs(self.wz - self.wz_target) < 1e-6,
                                         self.ang_hold_time, 'wz_hold_timer', 'wz_phase')

        # ===== 메시지 발행 =====
        msg = Twist()
        msg.linear.x  = self.v
        msg.angular.z = self.wz + self.ang_bias
        self.pub.publish(msg)

        self.get_logger().info(
            f'[LIN] phase={self.v_phase} v={self.v:+.3f}->{self.v_target:+.3f} (step={self.accel * self.dt:.3f}) | '
            f'[ANG] phase={self.wz_phase} wz={self.wz:+.3f}->{self.wz_target:+.3f} (step={self.ang_accel * self.dt:.3f})'
        )

def main():
    rclpy.init()
    node = RampCmdVelPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
