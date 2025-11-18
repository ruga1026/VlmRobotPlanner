#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Humble: Subscribe to /map and TF, overlay world origin + robot pose,
display a live OpenCV window, and (optionally) save PNGs. Adds same-gray padding
around the map so labels can extend outside the original area.

pip install opencv-python numpy
"""
import json
from std_msgs.msg import String
import math
from typing import Optional
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_image(xw, yw, origin_xy_yaw, resolution, img_height):
    """world (xw,yw)[m] -> image pixel (col,row) for ROS map (top-left image origin)"""
    x0, y0, yaw = origin_xy_yaw
    dx = xw - x0
    dy = yw - y0
    c = math.cos(yaw); s = math.sin(yaw)
    u_m = c * dx + s * dy
    v_m = -s * dx + c * dy
    u = u_m / resolution
    v = v_m / resolution
    col = u
    row = (img_height - 1) - v
    return float(col), float(row)


def occ_grid_to_gray_img(grid: OccupancyGrid) -> np.ndarray:
    """OccupancyGrid -> uint8 grayscale image. -1->205, 0->255, 100->0. Flipped vertically."""
    w = grid.info.width
    h = grid.info.height
    data = np.asarray(grid.data, dtype=np.int16).reshape((h, w))
    img = np.empty((h, w), dtype=np.uint8)
    unknown = data < 0
    img[unknown] = 205
    known = ~unknown
    clipped = np.clip(data[known], 0, 100)
    img[known] = (255 - (clipped * 255 // 100)).astype(np.uint8)
    return np.flipud(img)


class MapAnnotatorNode(Node):
    def __init__(self):
        super().__init__("map_annotator")

        # Parameters
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "camera_link")
        self.declare_parameter("save_path", "annotated_map.png")
        self.declare_parameter("save_period_sec", 0.5)
        self.declare_parameter("draw_origin", True)
        self.declare_parameter("arrow_len_m", 0.5)
        self.declare_parameter("text_scale", 0.4)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_scale", 1.0)
        self.declare_parameter("stale_after_sec", 5.0)
        # ⬇︎ 이제 이 값은 '언더레이(맵)에만' 적용됩니다.
        self.declare_parameter("render_scale", 3.0)
        # 오버레이 안티앨리어싱용 초해상도 배율(오버레이 두께/폰트에는 반영 X)
        self.declare_parameter("overlay_supersample", 2.0)
        # 동일 회색 패딩
        self.declare_parameter("pad_px", 5)
        # 화면에 표시할 로봇 이름 설정
        self.declare_parameter("robot_name", "robot")
        self.declare_parameter("publish_image", True)
        self.declare_parameter("image_pub_topic", "/merged_map")

        self.publish_image = bool(self.get_parameter("publish_image").value)
        self.image_pub_topic = self.get_parameter("image_pub_topic").get_parameter_value().string_value

        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.save_path = self.get_parameter("save_path").get_parameter_value().string_value
        self.save_period = float(self.get_parameter("save_period_sec").value)
        self.draw_origin = bool(self.get_parameter("draw_origin").value)
        self.arrow_len_m = float(self.get_parameter("arrow_len_m").value)
        self.text_scale = float(self.get_parameter("text_scale").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_scale = float(self.get_parameter("window_scale").value)
        self.stale_after = float(self.get_parameter("stale_after_sec").value)
        self.render_scale = float(self.get_parameter("render_scale").value)  # UNDERLAY ONLY
        self.overlay_supersample = float(self.get_parameter("overlay_supersample").value)
        self.pad_px = int(self.get_parameter("pad_px").value)
        self.robot_name = self.get_parameter("robot_name").get_parameter_value().string_value

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Map sub (latched)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_msg: Optional[OccupancyGrid] = None
        self.last_map_time: Optional[Time] = None
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, qos)

        # Detected objects (JSON)
        self.detected_objects = []
        obj_qos = QoSProfile(depth=10,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
        self.obj_sub = self.create_subscription(String, "/detected_objects_json",
                                                self.on_objects_json, obj_qos)
        self.bridge = CvBridge()
        img_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.image_pub = self.create_publisher(Image, self.image_pub_topic, img_qos)

        self.timer = self.create_timer(self.save_period, self.on_timer)
        self.window_name = "MapAnnotator (Live)"
        self.window_enabled = self.show_window

        self.get_logger().info(
            f"map_topic={self.map_topic}, base_frame={self.base_frame}, "
            f"render_scale(underlay only)={self.render_scale}, pad_px={self.pad_px}"
        )

    def on_map(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.last_map_time = self.get_clock().now()

    def get_robot_pose(self):
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time(), timeout=Duration(seconds=0.25)
            )
            tr = t.transform.translation
            q = t.transform.rotation
            return float(tr.x), float(tr.y), float(yaw_from_quat(q))
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({self.map_frame}->{self.base_frame}): {e}")
            return None

    def annotate(self, gray_img: np.ndarray, grid: OccupancyGrid, status_lines=None) -> np.ndarray:
        """
        UNDERLAY(맵)에는 render_scale을 적용, OVERLAY(텍스트/마커)는 고정 두께/폰트로 그린다.
        """
        h, w = gray_img.shape[:2]
        pad = max(0, self.pad_px)
        R = max(0.01, float(self.render_scale))  # underlay-only scale
        S = max(1.0, float(self.overlay_supersample))  # overlay SSAA only
        border_th = max(1, int(round(2 * S)))

        # --- UNDERLAY: base map -> pad -> scale to (out_wR, out_hR) ---
        bgr_base = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        bg_gray = int(gray_img[0, 0]) if gray_img.size else 205
        bgr_padded = cv2.copyMakeBorder(
            bgr_base, pad, pad, pad, pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(bg_gray, bg_gray, bg_gray)
        )
        out_w, out_h = (w + 2 * pad), (h + 2 * pad)
        out_wR, out_hR = int(round(out_w * R)), int(round(out_h * R))
        bgr_underlay = cv2.resize(
            bgr_padded, (out_wR, out_hR),
            interpolation=(cv2.INTER_NEAREST if R >= 1.0 else cv2.INTER_AREA)
        )

        # --- OVERLAY: draw on a high-res canvas (R * S), then downsample to (out_wR, out_hR) ---
        W_hi, H_hi = int(round(out_w * R * S)), int(round(out_h * R * S))
        overlay_hi = np.zeros((H_hi, W_hi, 3), dtype=np.uint8)
        alpha_hi = np.zeros((H_hi, W_hi), dtype=np.uint8)

        # helpers: scale coordinates (map px -> on-screen px), with padding & underlay scale
        def _toX(x):  # col -> screen x
            return int(round((pad + x) * R * S))
        def _toY(y):  # row -> screen y
            return int(round((pad + y) * R * S))

        # overlay thickness/font: DO NOT depend on R (map scale). Only SSAA S affects them.
        th1 = max(1, int(round(1 * S)))
        th2 = max(1, int(round(2 * S)))
        font_scale = self.text_scale * S

        # grid origin pose
        origin_pose = grid.info.origin
        origin = (float(origin_pose.position.x),
                  float(origin_pose.position.y),
                  yaw_from_quat(origin_pose.orientation))
        res = float(grid.info.resolution)

        # --- draw world origin ---
        if self.draw_origin:
            col_f, row_f = world_to_image(0.0, 0.0, origin, res, h)
            ci, ri = int(round(col_f)), int(round(row_f))
            if 0 <= ci < w and 0 <= ri < h:
                x1, y1 = _toX(max(0, ci - 40)), _toY(ri)
                x2, y2 = _toX(ci), _toY(ri)
                # cv2.arrowedLine(overlay_hi, (x1, y1), (x2, y2), (0, 0, 255), th2, cv2.LINE_AA, 0, 0.25)
                # cv2.arrowedLine(alpha_hi, (x1, y1), (x2, y2), 255, th2, cv2.LINE_AA, 0, 0.25)
                cv2.circle(overlay_hi, (x2, y2), max(2, int(round(3 * S))), (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(alpha_hi, (x2, y2), max(2, int(round(3 * S))), 255, -1, cv2.LINE_AA)
                # loc = (_toX(min(ci + 6, w - 1)), _toY(max(ri - 10, 0)))
                # 변경: 점의 화면좌표 (x2,y2) 기준으로 화면 픽셀 오프셋
                dx = int(round(6 * S))   # 오른쪽으로 6*S px
                dy = int(round(10 * S))  # 위로 10*S px
                loc = (min(x2 + dx, W_hi - 1), max(y2 - dy, 0))
                # 텍스트 내용
                text = "base"
                # 텍스트 크기 계산
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, th1)
                # 좌표 (loc는 이미 정의됨)
                x, y = loc
                # 사각형 좌표 (텍스트를 감싸도록 패딩 추가)
                rect_tl = (x - 1, y - text_h - 2)         # 좌상단
                rect_br = (x + text_w + 2, y + baseline + 2)  # 우하단
                # 흰색 사각형 그리기
                cv2.rectangle(overlay_hi, rect_tl, rect_br, (255,255,255), border_th, cv2.LINE_AA)
                cv2.rectangle(alpha_hi,   rect_tl, rect_br, 255,           border_th, cv2.LINE_AA)
                cv2.putText(overlay_hi, "base", loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), th1, cv2.LINE_AA)
                cv2.putText(alpha_hi, "base", loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, th1, cv2.LINE_AA)
                # 텍스트 출력


        # --- draw robot ---
        rp = self.get_robot_pose()
        if rp is None:
            if status_lines is not None:
                status_lines.append(f"TF: {self.map_frame}->{self.base_frame} UNAVAILABLE")
        else:
            xw, yw, yaw_r = rp
            col_r, row_r = world_to_image(xw, yw, origin, res, h)
            cri, rri = int(round(col_r)), int(round(row_r))
            ahead = self.arrow_len_m
            xh = xw + ahead * math.cos(yaw_r)
            yh = yw + ahead * math.sin(yaw_r)
            col_h, row_h = world_to_image(xh, yh, origin, res, h)

            if 0 <= cri < w and 0 <= rri < h:
                xr, yr = _toX(cri), _toY(rri)
                xh2, yh2 = _toX(int(round(col_h))), _toY(int(round(row_h)))
                cv2.circle(overlay_hi, (xr, yr), max(2, int(round(4 * S))), (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(alpha_hi, (xr, yr), max(2, int(round(4 * S))), 255, -1, cv2.LINE_AA)
                cv2.arrowedLine(overlay_hi, (xr, yr), (xh2, yh2), (0, 255, 0), th2, cv2.LINE_AA, 0, 0.25)
                cv2.arrowedLine(alpha_hi, (xr, yr), (xh2, yh2), 255, th2, cv2.LINE_AA, 0, 0.25)
                label = f"{self.robot_name}"     #({xw:.2f},{yw:.2f})"
                # loc = (_toX(min(cri + 6, w - 1)), _toY(max(rri - 10, 0)))
                # 변경: 로봇 점의 화면좌표 (xr,yr) 기준으로 화면 픽셀 오프셋
                dx = int(round(6 * S))
                dy = int(round(10 * S))
                loc = (min(xr + dx, W_hi - 1), max(yr - dy, 0))
                loc_w, loc_h = loc
                # 텍스트 크기 계산
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, th1)
                # 좌표 (loc는 이미 정의됨)
                x, y = loc
                rect_tl = (max(0, x - 5), max(0, y - text_h - 6))
                rect_br = (min(W_hi - 1, x + text_w + 5), min(H_hi - 1, y + baseline + 4))                
                # x, y = loc
                # # 사각형 좌표 (텍스트를 감싸도록 패딩 추가)
                # rect_tl = (x - 1, y - text_h - 2)         # 좌상단
                # rect_br = (x + text_w + 2, y + baseline + 2)  # 우하단
                # 흰색 사각형 그리기
                cv2.rectangle(overlay_hi, rect_tl, rect_br, (255,255,255), border_th, cv2.LINE_AA)
                cv2.rectangle(alpha_hi,   rect_tl, rect_br, 255,           border_th, cv2.LINE_AA)
                cv2.putText(overlay_hi, label, loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), th1, cv2.LINE_AA)
                cv2.putText(alpha_hi, label, loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, th1, cv2.LINE_AA)
            else:
                self.get_logger().warn(f"Robot projects outside image: (col={col_r:.1f}, row={row_r:.1f})")

        # --- draw detected objects ---
        if self.detected_objects:
            for obj in self.detected_objects:
                xw, yw = obj["x"], obj["y"]
                col_o, row_o = world_to_image(xw, yw, origin, res, h)
                ci, ri = int(round(col_o)), int(round(row_o))
                if 0 <= ci < w and 0 <= ri < h:
                    color = self._color_for_class(obj.get("cls", ""))
                    xo, yo = _toX(ci), _toY(ri)
                    cv2.circle(overlay_hi, (xo, yo), max(2, int(round(4 * S))), color, -1, cv2.LINE_AA)
                    cv2.circle(alpha_hi, (xo, yo), max(2, int(round(4 * S))), 255, -1, cv2.LINE_AA)
                    label = f'{obj.get("id","") or obj.get("cls","obj")}' #({xw:.2f},{yw:.2f})
                    # loc = (_toX(min(ci + 6, w - 1)), _toY(max(ri - 10, 0)))
                    dx = int(round(6 * S))
                    dy = int(round(10 * S))
                    loc = (min(xo + dx, W_hi - 1), max(yo - dy, 0))
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, th1)
                    # 좌표 (loc는 이미 정의됨)
                    # x, y = loc
                    # # 사각형 좌표 (텍스트를 감싸도록 패딩 추가)
                    # rect_tl = (x - 1, y - text_h - 2)         # 좌상단
                    # rect_br = (x + text_w + 2, y + baseline + 2)  # 우하단
                    x, y = loc
                    rect_tl = (max(0, x - 5), max(0, y - text_h - 6))
                    rect_br = (min(W_hi - 1, x + text_w + 5), min(H_hi - 1, y + baseline + 4))

                    # 흰색 사각형 그리기
                    cv2.rectangle(overlay_hi, rect_tl, rect_br, (255,255,255), border_th, cv2.LINE_AA)
                    cv2.rectangle(alpha_hi,   rect_tl, rect_br, 255,           border_th, cv2.LINE_AA)
                    cv2.putText(overlay_hi, label, loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, th1, cv2.LINE_AA)
                    cv2.putText(alpha_hi, label, loc, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, th1, cv2.LINE_AA)

        # --- downsample overlay from (R*S) to (R) and alpha blend onto underlay ---
        overlay = cv2.resize(overlay_hi, (out_wR, out_hR), interpolation=cv2.INTER_AREA) if S > 1.0 else overlay_hi
        alpha = cv2.resize(alpha_hi, (out_wR, out_hR), interpolation=cv2.INTER_AREA) if S > 1.0 else alpha_hi
        alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]
        bgr = (overlay.astype(np.float32) * alpha_f +
               bgr_underlay.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)

        if status_lines:
            self._overlay_status_lines(bgr, status_lines)

        return bgr

    def on_timer(self):
        status = []
        now = self.get_clock().now()

        if self.map_msg is None:
            placeholder = self._blank_frame(640, 480)
            status.append(f"Waiting for map on {self.map_topic} …")
            status.append("Ensure map_server is running (transient local).")
            bgr = self._overlay_status_lines(placeholder, status)
            # self._maybe_show(bgr)
            return

        # if self.last_map_time is not None:
        #     age = (now - self.last_map_time).nanoseconds * 1e-9
        #     if age > self.stale_after:
        #         status.append(f"Map STALE: {age:.1f}s since last update")

        try:
            gray = occ_grid_to_gray_img(self.map_msg)
            annotated = self.annotate(gray, self.map_msg, status_lines=status)
            
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.map_frame  # 필요시 "map"
            self.image_pub.publish(msg)

            # ⬇︎ 이제 최종 이미지에 대한 추가 스케일링은 하지 않습니다.
            #    (render_scale은 언더레이에만 적용 완료)
            # self._maybe_show(annotated)

            # 필요시 저장(주석 해제)
            # cv2.imwrite(self.save_path, annotated)
        except Exception as e:
            self.get_logger().error(f"Failed to render/save: {e}")

    def _maybe_show(self, bgr_img: np.ndarray):
        if not self.window_enabled:
            return
        try:
            disp = bgr_img
            if self.window_scale != 1.0:
                disp = cv2.resize(
                    bgr_img,
                    (int(round(bgr_img.shape[1] * self.window_scale)),
                     int(round(bgr_img.shape[0] * self.window_scale))),
                    interpolation=cv2.INTER_LINEAR
                )
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self.get_logger().info("Exit requested from window. Shutting down…")
                rclpy.shutdown()
        except cv2.error as e:
            self.window_enabled = False
            self.get_logger().warn(
                f"OpenCV GUI unavailable; disabling window. Set show_window:=False. Error: {e}"
            )

    def _overlay_status_lines(self, img: np.ndarray, lines):
        x, y = 10, 20
        for line in lines:
            cv2.putText(img, line, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (0, 255, 255), 1, cv2.LINE_AA)
            y += int(22 * self.text_scale + 8)
        return img

    @staticmethod
    def _blank_frame(w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        return img

    def on_objects_json(self, msg: String):
        """Parse std_msgs/String JSON and store flat objects list with (x,y) in map frame."""
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Invalid JSON on /detected_objects_json: {e}")
            return

        frames = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
        new_list = []
        for fr in frames:
            objs = fr.get("objects", [])
            if not isinstance(objs, list):
                continue
            for o in objs:
                try:
                    oid = str(o.get("id", ""))
                    cls = str(o.get("class", ""))
                    xy = o.get("map_xy", None)
                    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                        new_list.append({"id": oid, "cls": cls, "x": float(xy[0]), "y": float(xy[1])})
                except Exception:
                    continue
        self.detected_objects = new_list

    def _color_for_class(self, cls: str):
        palette = {
            "cap":    (255, 0, 0),     # BGR
            "bottle": (0, 165, 255),
        }
        return palette.get(cls.lower(), (255, 255, 0))


def main():
    rclpy.init()
    node = MapAnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
