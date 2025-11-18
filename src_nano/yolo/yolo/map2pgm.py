#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map2pgm.py
- 저장된 맵(YAML+PGM/PNG)을 불러와 YOLO 등에서 얻은 월드 좌표 라벨을 지도 위에 그려 새 PGM/PNG로 저장
- 또는 YAML 없이 /map(OccupancyGrid)을 라이브로 구독하여 즉석 변환 + 라벨 후 저장
- 저장 시 타임스탬프 기반 파일명 자동 생성(예: 08201704map.pgm / .yaml)

ROS 2 Humble 기준
"""

import os
import math
import time
import yaml
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from geometry_msgs.msg import PoseArray

# vision_msgs는 선택적. 없으면 PoseArray만 사용.
try:
    from vision_msgs.msg import Detection3DArray
    HAS_VISION_MSGS = True
except Exception:
    HAS_VISION_MSGS = False

from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def yaw_from_quat(q):
    """geometry_msgs/Quaternion -> yaw(rad)"""
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_pixel(x, y, origin, res, H):
    """
    OccupancyGrid 정보로 월드좌표(x,y) -> 픽셀(u,v)
    origin = [ox, oy, theta], res[m/pixel], 이미지 높이 H(상단 0)
    """
    ox, oy, theta = origin
    dx = x - ox
    dy = y - oy
    c = math.cos(-theta)
    s = math.sin(-theta)
    xm = c * dx - s * dy
    ym = s * dx + c * dy

    u = int(np.floor(xm / res))
    v_map = int(np.floor(ym / res))
    v = H - 1 - v_map  # 이미지 좌표(y 아래로 증가) 보정
    return u, v


def draw_point_label(gray, u, v, text, box_w=42, box_h=18, box_gray=255, text_gray=0):
    """
    회색 pgm/PNG 이미지(gray)에 픽셀(u,v) 기준으로 점 + 텍스트 배경박스 + 라벨 렌더링
    """
    H, W = gray.shape[:2]
    x1, y1 = max(0, u - 10), max(0, v - 10)
    x2, y2 = min(W - 1, u + 10), min(H - 1, v + 10)
    cv2.rectangle(gray, (x1, y1), (x2, y2), color=box_gray, thickness=1)

    tbx1, tby1 = min(W - 1, x2 + 3), max(0, v - box_h // 2)
    tbx2, tby2 = min(W - 1, tbx1 + box_w), min(H - 1, tby1 + box_h)
    cv2.rectangle(gray, (tbx1, tby1), (tbx2, tby2), color=box_gray, thickness=-1)
    cv2.putText(gray, text, (tbx1 + 3, tby2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_gray, 1, cv2.LINE_AA)


class MapAnnotator(Node):
    """
    두 모드 지원:
    1) 파일 모드: map_yaml 파라미터에 저장된 맵 YAML 경로를 주면, 해당 맵을 로드해 라벨을 그리고 새 파일 저장
    2) 라이브 모드: map_yaml을 비워두면 /map(OccupancyGrid)을 구독하여 즉석 변환 후 라벨 + 저장
    입력:
      - vision_msgs/Detection3DArray (선택): class_id/score/pose 사용
      - geometry_msgs/PoseArray (선택): 좌표만 있을 때 라벨 prefix+index로 표기
    저장:
      - Trigger 서비스(/save_annotated_map)로 수동 저장
      - auto_save_period_sec > 0 이면 주기 저장
    """

    def __init__(self):
        super().__init__("map_annotator")

        # ===== 기본 파라미터 =====
        self.declare_parameter("map_yaml", "")                 # 비우면 라이브 /map 모드
        self.declare_parameter("output_basename", "annot_map") # 파일 모드일 때 기본 stem (미사용: timestamp 기반으로 대체)
        self.declare_parameter("image_format", "pgm")          # 'pgm' 또는 'png'

        self.declare_parameter("objects_topic", "/detections_3d")
        self.declare_parameter("poses_topic", "")  # 비우면 사용 안 함
        self.declare_parameter("label_prefix_for_posearray", "obj")
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("stale_sec", 10.0)              # n초 지나면 오래된 검출 삭제
        self.declare_parameter("auto_save_period_sec", 0.0)    # >0이면 주기 저장

        # ===== 파일명 자동화 파라미터 =====
        self.declare_parameter("timestamp_format", "%m%d%H%M") # 예: 08201704
        self.declare_parameter("name_suffix", "map")           # 예: 'map' -> 08201704map
        self.declare_parameter("output_dir", "")               # 비우면 현재 폴더

        # ===== 라이브 /map 구독 관련 =====
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_subscribe_transient_local", True)
        self.declare_parameter("free_thresh", 0.25)            # trinary 변환 문턱
        self.declare_parameter("occupied_thresh", 0.65)        # trinary 변환 문턱

        # 파라미터 읽기
        self.map_yaml = self.get_parameter("map_yaml").value
        self.output_basename = self.get_parameter("output_basename").value
        self.image_format = self.get_parameter("image_format").value.lower()
        self.objects_topic = self.get_parameter("objects_topic").value
        self.poses_topic = self.get_parameter("poses_topic").value
        self.label_prefix_pose = self.get_parameter("label_prefix_for_posearray").value
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.stale_sec = float(self.get_parameter("stale_sec").value)
        self.auto_save_period_sec = float(self.get_parameter("auto_save_period_sec").value)

        # 내부 상태
        self.base_gray = None     # 원본 맵 그레이스케일
        self.H = self.W = None
        self.resolution = None
        self.origin = [0.0, 0.0, 0.0]  # [x, y, theta]

        # 파일/라이브 모드 분기
        if self.map_yaml:
            self._load_map_from_yaml(self.map_yaml)
            self.live_mode = False
        else:
            self._subscribe_live_map()
            self.live_mode = True

        # 검출 캐시: dict(x=.., y=.., label=.., stamp=rclpy.time.Time)
        self.objects = []

        # 구독 설정
        if HAS_VISION_MSGS and self.objects_topic:
            self.sub_det3d = self.create_subscription(
                Detection3DArray, self.objects_topic, self.cb_det3d, 10
            )
            self.get_logger().info(f"Subscribed: Detection3DArray {self.objects_topic}")
        else:
            self.sub_det3d = None
            if not HAS_VISION_MSGS and self.objects_topic:
                self.get_logger().warn("vision_msgs 미설치: Detection3DArray 구독 비활성화(PoseArray만 사용).")

        if self.poses_topic:
            self.sub_posearr = self.create_subscription(
                PoseArray, self.poses_topic, self.cb_posearray, 10
            )
            self.get_logger().info(f"Subscribed: PoseArray {self.poses_topic}")
        else:
            self.sub_posearr = None

        # 저장 서비스
        self.srv = self.create_service(Trigger, "save_annotated_map", self.on_save_srv)

        # 주기 저장 타이머
        if self.auto_save_period_sec > 0.0:
            self.timer = self.create_timer(self.auto_save_period_sec, self.auto_save_tick)
            self.get_logger().info(f"Auto-save every {self.auto_save_period_sec:.1f}s enabled.")
        else:
            self.timer = None

        if self.live_mode:
            self.get_logger().info("Live /map mode enabled. OccupancyGrid 수신 대기 중...")
        else:
            self.get_logger().info("File mode enabled. 저장된 맵에서 후처리합니다.")

    # ---------- 맵 로딩/구독 ----------

    def _load_map_from_yaml(self, yaml_path: str):
        if not os.path.isfile(yaml_path):
            raise RuntimeError(f"map_yaml 파일을 찾을 수 없습니다: {yaml_path}")

        with open(yaml_path, "r") as f:
            self.map_info = yaml.safe_load(f)

        img_field = self.map_info.get("image")
        if img_field is None:
            raise RuntimeError("YAML에 'image' 항목이 없습니다.")

        if os.path.isabs(img_field):
            img_path = img_field
        else:
            img_path = os.path.join(os.path.dirname(yaml_path), img_field)

        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"맵 이미지 로드 실패: {img_path}")

        self.base_gray = gray
        self.H, self.W = gray.shape[:2]
        self.resolution = float(self.map_info["resolution"])
        # origin: [x, y, theta]
        org = self.map_info.get("origin", [0.0, 0.0, 0.0])
        if len(org) < 3:
            org = [float(org[0]), float(org[1]), 0.0]
        self.origin = [float(org[0]), float(org[1]), float(org[2])]

        self.get_logger().info(f"Loaded map: {img_path} (W={self.W}, H={self.H}, res={self.resolution}, origin={self.origin})")

    def _subscribe_live_map(self):
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if self.get_parameter("map_subscribe_transient_local").value else DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        topic = self.get_parameter("map_topic").value
        self.sub_map = self.create_subscription(OccupancyGrid, topic, self.cb_map, qos)
        self.map_info = None  # 라이브 모드에서는 YAML이 없으므로 저장 시 구성

    def cb_map(self, msg: OccupancyGrid):
        W = msg.info.width
        H = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        yaw = yaw_from_quat(msg.info.origin.orientation)

        data = np.array(msg.data, dtype=np.int16).reshape(H, W)  # -1, 0..100
        f_th = float(self.get_parameter("free_thresh").value)
        o_th = float(self.get_parameter("occupied_thresh").value)

        gray = np.empty((H, W), dtype=np.uint8)
        # unknown
        gray[data < 0] = 205
        # known normalize 0..100 -> 0..1
        known = data >= 0
        norm = np.zeros_like(data, dtype=np.float32)
        norm[known] = data[known] / 100.0
        # occupied=0(검정), free=254(흰), mid=205(회색)
        gray[np.logical_and(known, norm >= o_th)] = 0
        gray[np.logical_and(known, norm <= f_th)] = 254
        mid = np.logical_and(known, np.logical_and(norm > f_th, norm < o_th))
        gray[mid] = 205

        # ROS 맵 좌표(좌하단 원점) -> 이미지 좌표(좌상단 원점): 세로 뒤집기
        gray = np.flipud(gray)

        self.base_gray = gray
        self.H, self.W = H, W
        self.resolution = res
        self.origin = [ox, oy, yaw]

    # ---------- 입력 콜백 ----------

    def cb_det3d(self, msg: 'Detection3DArray'):
        now = self.get_clock().now()
        added = 0
        for det in msg.detections:
            class_id = ""
            score = None
            px = py = None

            # 결과(가설 + pose) 우선 사용
            if det.results:
                class_id = det.results[0].hypothesis.class_id
                score = det.results[0].hypothesis.score
                if (score is not None) and (score < self.min_confidence):
                    continue
                pose = det.results[0].pose.pose
                px = pose.position.x
                py = pose.position.y

            # 보조: bbox center 사용 가능
            if px is None or py is None:
                pose = det.bbox.center
                px = pose.position.x
                py = pose.position.y

            label = class_id if class_id else "obj"
            if score is not None:
                label = f"{label}_{score:.2f}"

            self.objects.append(dict(x=float(px), y=float(py), label=label, stamp=now))
            added += 1

        if added:
            self.get_logger().debug(f"Detections added: {added} (total {len(self.objects)})")
        self._purge_stale()

    def cb_posearray(self, msg: PoseArray):
        now = self.get_clock().now()
        added = 0
        for i, pose in enumerate(msg.poses):
            px = pose.position.x
            py = pose.position.y
            label = f"{self.label_prefix_pose}_{i+1}"
            self.objects.append(dict(x=float(px), y=float(py), label=label, stamp=now))
            added += 1

        if added:
            self.get_logger().debug(f"PoseArray added: {added} (total {len(self.objects)})")
        self._purge_stale()

    def _purge_stale(self):
        if self.stale_sec <= 0:
            return
        now = self.get_clock().now()
        keep = []
        for o in self.objects:
            age = (now - o["stamp"]).nanoseconds / 1e9
            if age <= self.stale_sec:
                keep.append(o)
        dropped = len(self.objects) - len(keep)
        self.objects = keep
        if dropped > 0:
            self.get_logger().debug(f"Stale dropped: {dropped}")

    # ---------- 저장 ----------

    def on_save_srv(self, request, response):
        ok, path = self._save_annotated_map()
        response.success = ok
        response.message = path if ok else "save failed"
        return response

    def auto_save_tick(self):
        ok, path = self._save_annotated_map()
        if ok:
            self.get_logger().info(f"Auto-saved: {path}")

    def _save_annotated_map(self):
        if self.base_gray is None or self.resolution is None:
            self.get_logger().warn("맵 이미지/정보가 아직 없습니다. (/map 수신 대기 또는 YAML 로드 확인)")
            return False, ""

        # 파일명 자동 생성
        ts = time.strftime(self.get_parameter("timestamp_format").value)
        name_suffix = self.get_parameter("name_suffix").value
        output_dir = self.get_parameter("output_dir").value or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        stem = f"{ts}{name_suffix}"  # 예: 08201704 + map -> 08201704map
        ext = self.image_format if self.image_format in ("pgm", "png") else "png"
        out_img = os.path.join(output_dir, f"{stem}.{ext}")
        out_yaml = os.path.join(output_dir, f"{stem}.yaml")

        # 라벨 렌더링
        gray = self.base_gray.copy()
        count = 0
        for o in self.objects:
            u, v = world_to_pixel(o["x"], o["y"], self.origin, self.resolution, self.H)
            if 0 <= u < self.W and 0 <= v < self.H:
                draw_point_label(gray, u, v, o["label"])
                count += 1

        ok = cv2.imwrite(out_img, gray)
        if not ok:
            self.get_logger().error(f"이미지 저장 실패: {out_img}")
            return False, ""

        # YAML 저장
        if hasattr(self, "map_info") and self.map_info:
            # 파일 모드: 원본 YAML을 복사, image만 새 파일명으로 교체
            info = dict(self.map_info)
            info["image"] = os.path.basename(out_img)
            # origin/resolution 등은 그대로 유지
        else:
            # 라이브 모드: YAML을 새로 구성
            info = {
                "image": os.path.basename(out_img),
                "mode": "trinary",
                "resolution": float(self.resolution),
                "origin": [float(self.origin[0]), float(self.origin[1]), float(self.origin[2])],
                "negate": 0,
                "occupied_thresh": float(self.get_parameter("occupied_thresh").value),
                "free_thresh": float(self.get_parameter("free_thresh").value),
            }

        with open(out_yaml, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False)

        self.get_logger().info(f"Saved annotated map: {out_img} (labels={count}) & {out_yaml}")
        return True, os.path.abspath(out_img)


def main():
    rclpy.init()
    node = MapAnnotator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()