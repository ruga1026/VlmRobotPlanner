# import os
# from collections import deque, defaultdict
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
# import tf2_ros
# from sensor_msgs.msg import CameraInfo, Image
# from std_msgs.msg import String
# from visualization_msgs.msg import Marker
# from geometry_msgs.msg import PointStamped
# from cv_bridge import CvBridge
# from ament_index_python.packages import get_package_share_directory
# from ultralytics import YOLOE


# def parse_prompts(text: str):
#     """쉼표/공백 기준 분리 + 중복 제거(순서 유지)"""
#     if not text:
#         return []
#     tokens = text.replace(',', ' ').split()
#     seen = set()
#     out = []
#     for t in tokens:
#         t = t.strip()
#         if t and t not in seen:
#             out.append(t)
#             seen.add(t)
#     return out

# def _center_from_bbox(x1: float, y1: float, x2: float, y2: float):
#     """바운딩박스 중심(u, v) 계산"""
#     return ( (float(x1) + float(x2)) / 2.0,
#              (float(y1) + float(y2)) / 2.0 )

# class YoloMapMarkerMulti(Node):
#     def __init__(self):
#         super().__init__('yolo_map_marker_multi')
#         self.bridge = CvBridge()

#         # --- parameters ---
#         self.declare_parameter('map_resolution', 0.05)           # m/pixel
#         self.declare_parameter('use_latest_tf', True)            # Time(0) 조회
#         self.declare_parameter('match_radius', 0.20)             # 트랙-검출 매칭 반경(m)
#         self.declare_parameter('track_ttl', 3.0)                 # 마지막 관측 후 보존(s)
#         self.declare_parameter('conf_thresh', 0.15)

#         # 히트 창: 2초에 4회 (요청 반영)
#         self.declare_parameter('hit_window_sec', 2.0)
#         self.declare_parameter('hit_needed', 4)

#         self.declare_parameter('depth_log_delta', 0.05)          # Z 로그 임계(5cm)

#         # 앵커/스무딩
#         self.declare_parameter('lock_radius', 0.25)
#         self.declare_parameter('update_deadband', 0.03)
#         self.declare_parameter('smooth_alpha', 0.2)

#         # 재결합/상한/시각화
#         self.declare_parameter('anchor_reassoc_radius', 0.45)
#         self.declare_parameter('max_tracks_per_label', 6)
#         self.declare_parameter('show_radii', True)

#         # 깊이 보정(보정1: 상하좌우 평균만 사용; r=1이면 중심+상하좌우 1픽셀)
#         self.declare_parameter('depth_fallback_radius', 1)

#         # 디버그/마커
#         self.declare_parameter('marker_lifetime_sec', 1.0)       # 유령 마커 방지(0=영구)
#         self.declare_parameter('log_center_uv_only', True)       # 중심 (u,v)만 로그
#         self.declare_parameter('publish_center_uv', False)       # 필요 시 /center_uv 발행
#         self.declare_parameter('center_mode', 'mask_median')     # 또는 'bbox_center'

#         # --- load params ---
#         self.map_res = float(self.get_parameter('map_resolution').value)
#         self.use_latest_tf = bool(self.get_parameter('use_latest_tf').value)
#         self.match_radius = float(self.get_parameter('match_radius').value)
#         self.track_ttl = float(self.get_parameter('track_ttl').value)
#         self.conf_thresh = float(self.get_parameter('conf_thresh').value)
#         self.hit_window = Duration(seconds=float(self.get_parameter('hit_window_sec').value))
#         self.hit_needed = int(self.get_parameter('hit_needed').value)
#         self.depth_log_delta = float(self.get_parameter('depth_log_delta').value)

#         self.lock_radius = float(self.get_parameter('lock_radius').value)
#         self.update_deadband = float(self.get_parameter('update_deadband').value)
#         self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)

#         self.anchor_reassoc_radius = float(self.get_parameter('anchor_reassoc_radius').value)
#         self.max_tracks_per_label = int(self.get_parameter('max_tracks_per_label').value)
#         self.show_radii = bool(self.get_parameter('show_radii').value)

#         self.depth_fb_r = int(self.get_parameter('depth_fallback_radius').value)

#         self.marker_lifetime = float(self.get_parameter('marker_lifetime_sec').value)
#         self.log_center_uv_only = bool(self.get_parameter('log_center_uv_only').value)
#         self.publish_center_uv = bool(self.get_parameter('publish_center_uv').value)
#         self.center_mode = str(self.get_parameter('center_mode').value).strip().lower()

#         self.get_logger().info(
#             f"Radii -> match:{self.match_radius:.2f} m, lock:{self.lock_radius:.2f} m, "
#             f"reassoc:{self.anchor_reassoc_radius:.2f} m, deadband:{self.update_deadband:.2f} m"
#         )
#         self.get_logger().info(
#             f"Hit policy -> window:{float(self.get_parameter('hit_window_sec').value):.1f}s, "
#             f"needed:{self.hit_needed}"
#         )
#         self.get_logger().info(
#             f"Depth policy -> center-only, fallback=cross_mean, radius={self.depth_fb_r}"
#         )

#         # --- subs/pubs ---
#         self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, 10)
#         self.create_subscription(Image, '/camera/color/image_raw', self.rgb_cb, 10)
#         self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)
#         self.create_subscription(String, '/typed_input', self.text_cb, 10)

#         qos = QoSProfile(depth=1)
#         qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
#         qos.reliability = QoSReliabilityPolicy.RELIABLE
#         self.marker_pub = self.create_publisher(Marker, 'detected_object', qos)

#         self.uv_pub = self.create_publisher(PointStamped, 'center_uv', 10) if self.publish_center_uv else None

#         # --- TF ---
#         self.tfbuf = tf2_ros.Buffer()
#         self.tflistener = tf2_ros.TransformListener(self.tfbuf, self)

#         # --- YOLOE ---
#         model_path = os.path.join(get_package_share_directory('yolo'), 'yoloe-11s-seg.pt')
#         if not os.path.exists(model_path):
#             self.get_logger().error(f"모델 파일 없음: {model_path}")
#             raise FileNotFoundError(model_path)
#         self.model = YOLOE(model_path, task='segment')
#         self.get_logger().info(f"YOLOE 모델 로드 완료: {model_path}")

#         # --- state ---
#         self.K = None
#         self.cam_w = None
#         self.cam_h = None
#         self.prompts = []
#         self.latest_rgb = None
#         self.latest_depth = None

#         self.conf_hits = defaultdict(deque)   # label -> deque[timestamps]
#         self.tracks = []
#         self.next_track_id = 1

#         # --- pre-hit buffer: label -> deque of {u,v,Z,P_cam,stamp,rcv_time} ---
#         self.pre_hit_buf = defaultdict(lambda: deque(maxlen=200))  # 충분히 큰 버퍼

#         self.create_timer(0.2, self.timer_cb)   # 5 Hz

#     # --------- callbacks ----------
#     def info_cb(self, msg: CameraInfo):
#         if self.K is None:
#             self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
#             self.cam_w = int(msg.width)
#             self.cam_h = int(msg.height)
#             self.get_logger().info(f"Camera intrinsics loaded: {self.cam_w}x{self.cam_h}")

#     def text_cb(self, msg: String):
#         prompts = parse_prompts(msg.data)
#         self.prompts = prompts
#         self.conf_hits.clear()
#         self.pre_hit_buf.clear()  # 프롬프트 바뀌면 이전 버퍼 폐기
#         if self.prompts:
#             pe = self.model.get_text_pe(self.prompts)
#             self.model.set_classes(self.prompts, pe)
#             self.get_logger().info(f"Prompts set: {', '.join(self.prompts)}")
#             self.tracks.clear()
#             self.next_track_id = 1
#         else:
#             self.get_logger().info("Prompts cleared.")
#             self.tracks.clear()

#     def rgb_cb(self, msg: Image):
#         self.latest_rgb = msg

#     def depth_cb(self, msg: Image):
#         self.latest_depth = msg

#     # --------- helpers ----------
#     def quaternion_to_matrix(self, q):
#         x, y, z, w = q.x, q.y, q.z, q.w
#         return np.array([
#             [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
#             [2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
#             [2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)]
#         ], dtype=np.float64)

#     def cam_to_map(self, p_cam, tf_cam2map):
#         q = tf_cam2map.transform.rotation
#         t = tf_cam2map.transform.translation
#         R = self.quaternion_to_matrix(q)
#         o = np.array([t.x, t.y, t.z], dtype=np.float64)
#         return R @ p_cam + o

#     # 중심 한 점 깊이
#     def depth_at_center(self, depth_img, u, v):
#         h, w = depth_img.shape
#         ui = int(round(u)); vi = int(round(v))
#         if ui < 0 or ui >= w or vi < 0 or vi >= h:
#             return np.nan
#         return float(depth_img[vi, ui])

#     # 보정1: 상하좌우 평균
#     def depth_fallback_cross_mean(self, depth_img, u, v, r):
#         if r <= 0:
#             return np.nan
#         h, w = depth_img.shape
#         ui = int(round(u)); vi = int(round(v))
#         vals = []
#         def grab(x, y):
#             if 0 <= x < w and 0 <= y < h:
#                 z = float(depth_img[y, x])
#                 if np.isfinite(z) and 0.05 < z < 10.0:
#                     vals.append(z)
#         grab(ui, vi)
#         for d in range(1, r+1):
#             grab(ui, vi - d); grab(ui, vi + d)
#             grab(ui - d, vi); grab(ui + d, vi)
#         return float(np.mean(vals)) if vals else np.nan

#     def depth_at_center_with_fallback(self, depth_img, u, v):
#         Z = self.depth_at_center(depth_img, u, v)
#         if np.isfinite(Z) and 0.05 < Z < 10.0:
#             return Z
#         return self.depth_fallback_cross_mean(depth_img, u, v, self.depth_fb_r)

#     def _lookup_tf(self, src_frame, stamp):
#         try:
#             if self.use_latest_tf:
#                 return self.tfbuf.lookup_transform('map', src_frame, rclpy.time.Time(),
#                                                    timeout=Duration(seconds=0.2))
#             if self.tfbuf.can_transform('map', src_frame, stamp, timeout=Duration(seconds=0.5)):
#                 return self.tfbuf.lookup_transform('map', src_frame, stamp, timeout=Duration(seconds=0.2))
#             return self.tfbuf.lookup_transform('map', src_frame, rclpy.time.Time(),
#                                                timeout=Duration(seconds=0.2))
#         except Exception:
#             return None

#     def _shrink_window(self, label):
#         now = self.get_clock().now()
#         dq = self.conf_hits[label]
#         while dq and (now - dq[0]) > self.hit_window:
#             dq.popleft()

#     def _shrink_pre_hit(self, label):
#         """hit_window 바깥의 pre-hit 항목 제거"""
#         now = self.get_clock().now()
#         dq = self.pre_hit_buf[label]
#         while dq and (now - dq[0]['rcv_time']) > self.hit_window:
#             dq.popleft()

#     def _hit(self, label):
#         now = self.get_clock().now()
#         dq = self.conf_hits[label]
#         dq.append(now)
#         self._shrink_window(label)
#         return len(dq) >= self.hit_needed

#     def _match_track(self, pos_map, label):
#         indices = [i for i, t in enumerate(self.tracks) if t['label'] == label]
#         if not indices:
#             return -1
#         dists = [np.linalg.norm(self.tracks[i]['pos'] - pos_map) for i in indices]
#         jloc = int(np.argmin(dists))
#         return indices[jloc] if dists[jloc] <= self.match_radius else -1

#     def _find_anchor_hit(self, label, p_map):
#         best_i, best_d = -1, 1e9
#         for i, t in enumerate(self.tracks):
#             if t['label'] != label:
#                 continue
#             d = np.linalg.norm(t['anchor'] - p_map)
#             if d < self.lock_radius and d < best_d:
#                 best_i, best_d = i, d
#         return best_i

#     def _match_anchor_radius(self, label, p_map):
#         best_i, best_d = -1, 1e9
#         for i, t in enumerate(self.tracks):
#             if t['label'] != label:
#                 continue
#             d = np.linalg.norm(t['anchor'] - p_map)
#             if d <= self.anchor_reassoc_radius and d < best_d:
#                 best_i, best_d = i, d
#         return best_i

#     def _count_label(self, label):
#         return sum(1 for t in self.tracks if t['label'] == label)

#     def _delete_markers(self, t, now):
#         # object dot
#         mk = Marker()
#         mk.header.frame_id = 'map'
#         mk.header.stamp = now.to_msg()
#         mk.ns = f"yolo_{t['label']}"
#         mk.id = int(t['id'])
#         mk.action = Marker.DELETE
#         self.marker_pub.publish(mk)
#         # lock sphere
#         mk2 = Marker()
#         mk2.header.frame_id = 'map'
#         mk2.header.stamp = now.to_msg()
#         mk2.ns = f"radius_lock_{t['label']}"
#         mk2.id = 10000 + int(t['id'])
#         mk2.action = Marker.DELETE
#         self.marker_pub.publish(mk2)
#         # reassoc sphere
#         mk3 = Marker()
#         mk3.header.frame_id = 'map'
#         mk3.header.stamp = now.to_msg()
#         mk3.ns = f"radius_reassoc_{t['label']}"
#         mk3.id = 20000 + int(t['id'])
#         mk3.action = Marker.DELETE
#         self.marker_pub.publish(mk3)

#     def _publish_radius_markers(self, t, label, stamp):
#         if not self.show_radii:
#             return
#         # lock_radius (연두)
#         mk = Marker()
#         mk.header.frame_id = 'map'
#         mk.header.stamp = stamp.to_msg()
#         mk.ns = f"radius_lock_{label}"
#         mk.id = 10000 + int(t['id'])
#         mk.type = Marker.SPHERE
#         mk.action = Marker.ADD
#         mk.pose.position.x = float(t['anchor'][0])
#         mk.pose.position.y = float(t['anchor'][1])
#         mk.pose.position.z = float(t['anchor'][2])
#         r = float(self.lock_radius)
#         mk.scale.x = mk.scale.y = mk.scale.z = 2.0 * r
#         mk.color.r = 0.3; mk.color.g = 1.0; mk.color.b = 0.3; mk.color.a = 0.15
#         mk.lifetime.sec = int(self.marker_lifetime)
#         mk.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
#         self.marker_pub.publish(mk)

#         # anchor_reassoc_radius (파랑)
#         mk2 = Marker()
#         mk2.header.frame_id = 'map'
#         mk2.header.stamp = stamp.to_msg()
#         mk2.ns = f"radius_reassoc_{label}"
#         mk2.id = 20000 + int(t['id'])
#         mk2.type = Marker.SPHERE
#         mk2.action = Marker.ADD
#         mk2.pose.position.x = float(t['anchor'][0])
#         mk2.pose.position.y = float(t['anchor'][1])
#         mk2.pose.position.z = float(t['anchor'][2])
#         r2 = float(self.anchor_reassoc_radius)
#         mk2.scale.x = mk2.scale.y = mk2.scale.z = 2.0 * r2
#         mk2.color.r = 0.3; mk2.color.g = 0.3; mk2.color.b = 1.0; mk2.color.a = 0.10
#         mk2.lifetime.sec = int(self.marker_lifetime)
#         mk2.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
#         self.marker_pub.publish(mk2)

#     def _prune_tracks(self, now):
#         alive = []
#         for t in self.tracks:
#             if (now - t['last_seen']).nanoseconds * 1e-9 <= self.track_ttl:
#                 alive.append(t)
#             else:
#                 # 삭제되는 트랙의 마커를 즉시 제거
#                 self._delete_markers(t, now)
#         self.tracks = alive

#     # --------- main timer (5 Hz) ----------
#     def timer_cb(self):
#         if self.K is None or not self.prompts:
#             return
#         if self.latest_rgb is None or self.latest_depth is None:
#             return

#         # OpenCV 변환 + 깊이 인코딩(m) 보정
#         try:
#             cv_img = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
#             depth_img = self.bridge.imgmsg_to_cv2(
#                 self.latest_depth, desired_encoding='passthrough'
#             ).astype(np.float32)
#             enc = (self.latest_depth.encoding or '').lower()
#             if '16uc1' in enc or 'z16' in enc or 'mono16' in enc:
#                 depth_img *= 0.001  # mm -> m
#             dh, dw = depth_img.shape
#             if (self.cam_w is not None and self.cam_h is not None) and (dw != self.cam_w or dh != self.cam_h):
#                 self.get_logger().warn(
#                     f"camera_info({self.cam_w}x{self.cam_h}) vs depth({dw}x{dh}) 불일치. 정렬 필요."
#                 )
#                 return
#         except Exception:
#             return

#         # YOLOE 추론
#         try:
#             results = self.model.predict(source=cv_img, prompts=self.prompts, verbose=False)
#         except Exception:
#             return
#         if len(results) == 0 or results[0].boxes is None:
#             for label in self.prompts:
#                 self._shrink_window(label)
#                 self._shrink_pre_hit(label)
#             return

#         boxes = results[0].boxes
#         confs = boxes.conf.detach().cpu().numpy()
#         if confs.size == 0:
#             for label in self.prompts:
#                 self._shrink_window(label)
#                 self._shrink_pre_hit(label)
#             return

#         xyxy_all = boxes.xyxy.detach().cpu().numpy()
#         if getattr(boxes, 'cls', None) is not None:
#             cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
#         else:
#             cls_arr = np.zeros((xyxy_all.shape[0],), dtype=int)

#         masks = None
#         if hasattr(results[0], 'masks') and results[0].masks is not None:
#             masks = results[0].masks.data.detach().cpu().numpy()  # [N,H,W]

#         # TF (없어도 '로그/깊이'는 진행)
#         src_frame = self.latest_depth.header.frame_id
#         stamp = self.latest_depth.header.stamp
#         tf = self._lookup_tf(src_frame, stamp)
#         if tf is None:
#             self.get_logger().warn("TF(map<-depth) 없음: 이번 프레임은 마커/맵 투영을 생략하고 로그만 기록함.")

#         now = self.get_clock().now()
#         self._prune_tracks(now)

#         fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
#         cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
#         dot = max(2.0 * self.map_res, 0.1)

#         # 유효 박스 반복
#         for i in range(xyxy_all.shape[0]):
#             conf = float(confs[i])
#             if conf < self.conf_thresh:
#                 continue

#             cls_idx = int(cls_arr[i])
#             label = self.prompts[cls_idx] if (0 <= cls_idx < len(self.prompts)) else self.prompts[0]

#             x1, y1, x2, y2 = xyxy_all[i]

#             u, v = _center_from_bbox(x1, y1, x2, y2)

#             # 깊이 Z (중심 한 점 + 보정1) - 히트/TF와 무관하게 먼저 측정 & 로그
#             Z = self.depth_at_center_with_fallback(depth_img, u, v)
#             if self.log_center_uv_only:
#                 if np.isfinite(Z) and 0.05 < Z < 10.0:
#                     self.get_logger().info(f"[center] {label}: u={u:.1f}, v={v:.1f}, Z={Z:.3f} m")
#                 else:
#                     self.get_logger().info(f"[center] {label}: u={u:.1f}, v={v:.1f}, Z=NaN")

#             # 유효 깊이가 아니면 이후 처리 생략 (버퍼에도 저장 안 함)
#             if not (np.isfinite(Z) and 0.05 < Z < 10.0):
#                 continue

#             # 3D (camera)로 미리 복원 (pre-hit도 맵 좌표화 준비)
#             X = (u - cx) * Z / fx
#             Y = (v - cy) * Z / fy
#             P_cam = np.array([X, Y, Z], dtype=np.float64)

#             # pre-hit 버퍼에 저장 (히트 전 프레임 임시 보관)
#             self.pre_hit_buf[label].append({
#                 'u': float(u),
#                 'v': float(v),
#                 'Z': float(Z),
#                 'P_cam': P_cam,
#                 'stamp': stamp,          # 현재 depth의 stamp 사용
#                 'rcv_time': now
#             })
#             self._shrink_pre_hit(label)

#             # (옵션) 중심 좌표 토픽 발행
#             if self.publish_center_uv and self.uv_pub is not None:
#                 ps = PointStamped()
#                 ps.header = self.latest_depth.header
#                 ps.point.x = float(u)
#                 ps.point.y = float(v)
#                 ps.point.z = float(Z)
#                 self.uv_pub.publish(ps)

#             # 히트 정책 체크 (로그/버퍼 저장은 이미 했으므로 아래서 판정)
#             if not self._hit(label):
#                 continue

#             # --- 여기부터 히트 통과: pre-hit 버퍼 활용해 트랙/마커 생성 ---
#             # TF가 현재 없으면, 버퍼의 항목 stamp로 재시도
#             use_tf = tf
#             cand = None
#             if len(self.pre_hit_buf[label]) > 0:
#                 cand = self.pre_hit_buf[label][-1]  # 가장 최근 항목 사용
#                 if use_tf is None:
#                     use_tf = self._lookup_tf(src_frame, cand['stamp'])

#             if use_tf is None:
#                 # TF 여전히 없으면 다음 라운드로 미룸 (버퍼 유지)
#                 continue

#             # 버퍼에서 후보 맵 좌표 구해 트랙 생성/갱신
#             if cand is not None:
#                 P_map_seed = self.cam_to_map(cand['P_cam'], use_tf)
#                 Z_seed = cand['Z']
#             else:
#                 # 이론상 올 일 없지만, 방어적으로 현 프레임으로 seed
#                 P_map_seed = self.cam_to_map(P_cam, use_tf)
#                 Z_seed = Z

#             # 기존 트랙 매칭
#             j = self._match_track(P_map_seed, label)
#             if j < 0:
#                 # 앵커 히트/재결합 검사
#                 k_lock = self._find_anchor_hit(label, P_map_seed)
#                 if k_lock >= 0:
#                     t = self.tracks[k_lock]
#                     t['last_seen'] = now
#                     if (t.get('last_depth_logged') is None) or abs(Z_seed - t['last_depth_logged']) >= self.depth_log_delta:
#                         self.get_logger().info(f"[{t['name']}] Depth={Z_seed:.3f} m")
#                         t['last_depth_logged'] = Z_seed
#                     self._publish_marker(t, label, now, dot)
#                     self._publish_radius_markers(t, label, now)
#                 else:
#                     # 새 트랙 생성 (seed 사용)
#                     gid = self.next_track_id
#                     tname = f"{label}_{gid}"
#                     self.next_track_id += 1
#                     self.tracks.append({
#                         'id': gid,
#                         'name': tname,
#                         'label': label,
#                         'pos': P_map_seed.copy(),
#                         'anchor': P_map_seed.copy(),
#                         'last_seen': now,
#                         'last_depth_logged': Z_seed
#                     })
#                     self.get_logger().info(f"[{tname}] Depth={Z_seed:.3f} m")
#                     t = self.tracks[-1]
#                     self._publish_marker(t, label, now, dot)
#                     self._publish_radius_markers(t, label, now)
#             else:
#                 # 기존 트랙 업데이트 (seed 위치 기준)
#                 t = self.tracks[j]
#                 t['last_seen'] = now
#                 dist_anchor = np.linalg.norm(P_map_seed - t['anchor'])
#                 if dist_anchor > self.lock_radius:
#                     dist_cur = np.linalg.norm(P_map_seed - t['pos'])
#                     if dist_cur > self.update_deadband:
#                         a = self.smooth_alpha
#                         t['pos'] = a * P_map_seed + (1.0 - a) * t['pos']
#                 if (t.get('last_depth_logged') is None) or abs(Z_seed - t['last_depth_logged']) >= self.depth_log_delta:
#                     self.get_logger().info(f"[{t['name']}] Depth={Z_seed:.3f} m")
#                     t['last_depth_logged'] = Z_seed
#                 self._publish_marker(t, label, now, dot)
#                 self._publish_radius_markers(t, label, now)

#             # 사용한 뒤에는 해당 라벨의 pre-hit 버퍼 비우기 (요청 정책)
#             self.pre_hit_buf[label].clear()

#     def _publish_marker(self, t, label, stamp, dot):
#         marker = Marker()
#         marker.header.frame_id = 'map'
#         marker.header.stamp = stamp.to_msg()
#         marker.ns = f"yolo_{label}"
#         marker.id = int(t['id'])
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         marker.pose.position.x = float(t['pos'][0])
#         marker.pose.position.y = float(t['pos'][1])
#         marker.pose.position.z = float(t['pos'][2])
#         marker.scale.x = marker.scale.y = marker.scale.z = float(dot)
#         marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0
#         marker.lifetime.sec = int(self.marker_lifetime)
#         marker.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
#         self.marker_pub.publish(marker)


# def main(args=None):
#     rclpy.init(args=args)
#     node = YoloMapMarkerMulti()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
    
#!/usr/bin/env python3
# #0821 수정전 코드
# import os
# from collections import deque, defaultdict
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
# import tf2_ros
# from sensor_msgs.msg import CameraInfo, Image
# from std_msgs.msg import String
# from visualization_msgs.msg import Marker
# from geometry_msgs.msg import PointStamped
# from cv_bridge import CvBridge
# from ament_index_python.packages import get_package_share_directory
# from ultralytics import YOLOE


# # --- utils -------------------------------------------------
# def parse_prompts(text: str):
#     """쉼표/공백 기준 분리 + 중복 제거(순서 유지) -> 앞 2개만 사용"""
#     if not text:
#         return []
#     tokens = text.replace(',', ' ').split()
#     seen, out = set(), []
#     for t in tokens:
#         t = t.strip()
#         if t and t not in seen:
#             out.append(t); seen.add(t)
#         if len(out) >= 2:  # 앞의 2개만
#             break
#     return out

# def center_from_bbox(x1: float, y1: float, x2: float, y2: float):
#     """바운딩박스 중심(u, v)"""
#     return ( (float(x1) + float(x2)) / 2.0,
#              (float(y1) + float(y2)) / 2.0 )

# def color_for_label(label: str):
#     """라벨 문자열을 고정 팔레트에 매핑(하드코딩 회피)"""
#     palette = [
#         (1.00, 0.40, 0.00),  # 주황
#         (0.20, 0.80, 0.20),  # 연녹
#         (0.20, 0.60, 1.00),  # 파랑
#         (0.80, 0.20, 0.80),  # 보라
#         (1.00, 0.70, 0.20),  # 살구
#         (0.20, 0.90, 0.90),  # 청록
#     ]
#     idx = abs(hash(label)) % len(palette)
#     return palette[idx]


# # --- node --------------------------------------------------
# class YoloMapMarkerMulti(Node):
#     def __init__(self):
#         super().__init__('yolo_map_marker_multi')
#         self.bridge = CvBridge()

#         # params (필요한 것만)
#         self.declare_parameter('map_resolution', 0.05)      # m/pixel
#         self.declare_parameter('use_latest_tf', True)       # Time(0) 조회
#         self.declare_parameter('match_radius', 0.20)        # 트랙-검출 매칭 반경(m)
#         self.declare_parameter('track_ttl', 3.0)            # 마지막 관측 후 보존(s)
#         self.declare_parameter('conf_thresh', 0.15)
#         self.declare_parameter('hit_window_sec', 2.0)       # 2초 창
#         self.declare_parameter('hit_needed', 4)             # 2초 내 4회 (바로 보려면 1로)
#         self.declare_parameter('depth_log_delta', 0.05)     # 깊이 로그 임계(5cm)
#         self.declare_parameter('lock_radius', 0.25)
#         self.declare_parameter('update_deadband', 0.03)
#         self.declare_parameter('smooth_alpha', 0.2)
#         self.declare_parameter('anchor_reassoc_radius', 0.45)
#         self.declare_parameter('max_tracks_per_label', 6)
#         self.declare_parameter('show_radii', True)
#         self.declare_parameter('marker_lifetime_sec', 1.0)
#         self.declare_parameter('log_center_uv_only', True)  # u,v,Z 로그
#         self.declare_parameter('publish_center_uv', False)  # /center_uv 발행

#         # load params
#         self.map_res = float(self.get_parameter('map_resolution').value)
#         self.use_latest_tf = bool(self.get_parameter('use_latest_tf').value)
#         self.match_radius = float(self.get_parameter('match_radius').value)
#         self.track_ttl = float(self.get_parameter('track_ttl').value)
#         self.conf_thresh = float(self.get_parameter('conf_thresh').value)
#         self.hit_window = Duration(seconds=float(self.get_parameter('hit_window_sec').value))
#         self.hit_needed = int(self.get_parameter('hit_needed').value)
#         self.depth_log_delta = float(self.get_parameter('depth_log_delta').value)
#         self.lock_radius = float(self.get_parameter('lock_radius').value)
#         self.update_deadband = float(self.get_parameter('update_deadband').value)
#         self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)
#         self.anchor_reassoc_radius = float(self.get_parameter('anchor_reassoc_radius').value)
#         self.max_tracks_per_label = int(self.get_parameter('max_tracks_per_label').value)
#         self.show_radii = bool(self.get_parameter('show_radii').value)
#         self.marker_lifetime = float(self.get_parameter('marker_lifetime_sec').value)
#         self.log_center_uv_only = bool(self.get_parameter('log_center_uv_only').value)
#         self.publish_center_uv = bool(self.get_parameter('publish_center_uv').value)

#         self.get_logger().info(
#             f"Radii -> match:{self.match_radius:.2f} m, lock:{self.lock_radius:.2f} m, "
#             f"reassoc:{self.anchor_reassoc_radius:.2f} m, deadband:{self.update_deadband:.2f} m"
#         )
#         self.get_logger().info(
#             f"Hit policy -> window:{float(self.get_parameter('hit_window_sec').value):.1f}s, needed:{self.hit_needed}"
#         )

#         # subs/pubs
#         self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, 10)
#         self.create_subscription(Image, '/camera/color/image_raw', self.rgb_cb, 10)
#         self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)
#         self.create_subscription(String, '/typed_input', self.text_cb, 10)

#         qos = QoSProfile(depth=1)
#         qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
#         qos.reliability = QoSReliabilityPolicy.RELIABLE
#         self.marker_pub = self.create_publisher(Marker, 'detected_object', qos)
#         self.uv_pub = self.create_publisher(PointStamped, 'center_uv', 10) if self.publish_center_uv else None

#         # TF
#         self.tfbuf = tf2_ros.Buffer()
#         self.tflistener = tf2_ros.TransformListener(self.tfbuf, self)

#         # YOLOE (seg 모델이지만 마스크는 사용 안 함)
#         model_path = os.path.join(get_package_share_directory('yolo'), 'yoloe-11s-seg.pt')
#         if not os.path.exists(model_path):
#             self.get_logger().error(f"모델 파일 없음: {model_path}")
#             raise FileNotFoundError(model_path)
#         self.model = YOLOE(model_path, task='segment')
#         self.get_logger().info(f"YOLOE 모델 로드 완료: {model_path}")

#         # state
#         self.K = None; self.cam_w = None; self.cam_h = None
#         self.prompts = []
#         self.latest_rgb = None; self.latest_depth = None
#         self.conf_hits = defaultdict(deque)  # label -> deque[timestamps]
#         self.tracks = []; self.next_track_id = 1

#         self.create_timer(0.2, self.timer_cb)  # 5 Hz

#     # -- callbacks --
#     def info_cb(self, msg: CameraInfo):
#         if self.K is None:
#             self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
#             self.cam_w = int(msg.width); self.cam_h = int(msg.height)
#             self.get_logger().info(f"Camera intrinsics loaded: {self.cam_w}x{self.cam_h}")

#     def text_cb(self, msg: String):
#         self.prompts = parse_prompts(msg.data)  # 앞 2개만
#         self.conf_hits.clear()
#         if self.prompts:
#             pe = self.model.get_text_pe(self.prompts)
#             self.model.set_classes(self.prompts, pe)
#             self.get_logger().info(f"Prompts set: {', '.join(self.prompts)}")
#             self.tracks.clear(); self.next_track_id = 1
#         else:
#             self.get_logger().info("Prompts cleared.")
#             self.tracks.clear()

#     def rgb_cb(self, msg: Image):  self.latest_rgb = msg
#     def depth_cb(self, msg: Image): self.latest_depth = msg

#     # -- helpers --
#     def quaternion_to_matrix(self, q):
#         x, y, z, w = q.x, q.y, q.z, q.w
#         return np.array([
#             [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
#             [2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
#             [2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)]
#         ], dtype=np.float64)

#     def cam_to_map(self, p_cam, tf_cam2map):
#         q = tf_cam2map.transform.rotation
#         t = tf_cam2map.transform.translation
#         R = self.quaternion_to_matrix(q)
#         o = np.array([t.x, t.y, t.z], dtype=np.float64)
#         return R @ p_cam + o

#     def depth_at_center(self, depth_img, u, v):
#         h, w = depth_img.shape
#         ui = int(round(u)); vi = int(round(v))
#         if ui < 0 or ui >= w or vi < 0 or vi >= h:
#             return np.nan
#         return float(depth_img[vi, ui])

#     def _lookup_tf(self, src_frame, stamp):
#         try:
#             if self.use_latest_tf:
#                 return self.tfbuf.lookup_transform('map', src_frame, rclpy.time.Time(),
#                                                    timeout=Duration(seconds=0.2))
#             if self.tfbuf.can_transform('map', src_frame, stamp, timeout=Duration(seconds=0.5)):
#                 return self.tfbuf.lookup_transform('map', src_frame, stamp, timeout=Duration(seconds=0.2))
#             return self.tfbuf.lookup_transform('map', src_frame, rclpy.time.Time(),
#                                                timeout=Duration(seconds=0.2))
#         except Exception:
#             return None

#     def _shrink_window(self, label):
#         now = self.get_clock().now()
#         dq = self.conf_hits[label]
#         while dq and (now - dq[0]) > self.hit_window:
#             dq.popleft()

#     def _hit(self, label):
#         now = self.get_clock().now()
#         dq = self.conf_hits[label]
#         dq.append(now)
#         self._shrink_window(label)
#         return len(dq) >= self.hit_needed

#     def _match_track(self, pos_map, label):
#         indices = [i for i, t in enumerate(self.tracks) if t['label'] == label]
#         if not indices:
#             return -1
#         dists = [np.linalg.norm(self.tracks[i]['pos'] - pos_map) for i in indices]
#         jloc = int(np.argmin(dists))
#         return indices[jloc] if dists[jloc] <= self.match_radius else -1

#     def _find_anchor_hit(self, label, p_map):
#         best_i, best_d = -1, 1e9
#         for i, t in enumerate(self.tracks):
#             if t['label'] != label: continue
#             d = np.linalg.norm(t['anchor'] - p_map)
#             if d < self.lock_radius and d < best_d:
#                 best_i, best_d = i, d
#         return best_i

#     def _match_anchor_radius(self, label, p_map):
#         best_i, best_d = -1, 1e9
#         for i, t in enumerate(self.tracks):
#             if t['label'] != label: continue
#             d = np.linalg.norm(t['anchor'] - p_map)
#             if d <= self.anchor_reassoc_radius and d < best_d:
#                 best_i, best_d = i, d
#         return best_i

#     def _count_label(self, label):
#         return sum(1 for t in self.tracks if t['label'] == label)

#     def _publish_radius_markers(self, t, label, stamp):
#         if not self.show_radii: return
#         # lock (연두)
#         mk = Marker()
#         mk.header.frame_id = 'map'
#         mk.header.stamp = stamp.to_msg()
#         mk.ns = f"radius_lock_{label}"
#         mk.id = 10000 + int(t['id'])
#         mk.type = Marker.SPHERE
#         mk.action = Marker.ADD
#         mk.pose.position.x = float(t['anchor'][0])
#         mk.pose.position.y = float(t['anchor'][1])
#         mk.pose.position.z = float(t['anchor'][2])
#         r = float(self.lock_radius)
#         mk.scale.x = mk.scale.y = mk.scale.z = 2.0 * r
#         mk.color.r = 0.3; mk.color.g = 1.0; mk.color.b = 0.3; mk.color.a = 0.15
#         mk.lifetime.sec = int(self.marker_lifetime)
#         mk.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
#         self.marker_pub.publish(mk)
#         # reassoc (파랑)
#         mk2 = Marker()
#         mk2.header.frame_id = 'map'
#         mk2.header.stamp = stamp.to_msg()
#         mk2.ns = f"radius_reassoc_{label}"
#         mk2.id = 20000 + int(t['id'])
#         mk2.type = Marker.SPHERE
#         mk2.action = Marker.ADD
#         mk2.pose.position.x = float(t['anchor'][0])
#         mk2.pose.position.y = float(t['anchor'][1])
#         mk2.pose.position.z = float(t['anchor'][2])
#         r2 = float(self.anchor_reassoc_radius)
#         mk2.scale.x = mk2.scale.y = mk2.scale.z = 2.0 * r2
#         mk2.color.r = 0.3; mk2.color.g = 0.3; mk2.color.b = 1.0; mk2.color.a = 0.10
#         mk2.lifetime.sec = int(self.marker_lifetime)
#         mk2.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
#         self.marker_pub.publish(mk2)

#     def _prune_tracks(self, now):
#         alive = []
#         for t in self.tracks:
#             if (now - t['last_seen']).nanoseconds * 1e-9 <= self.track_ttl:
#                 alive.append(t)
#         self.tracks = alive

#     # -- main loop (5 Hz) --
#     def timer_cb(self):
#         if self.K is None or not self.prompts: return
#         if self.latest_rgb is None or self.latest_depth is None: return

#         # cv/depth
#         try:
#             cv_img = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
#             depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough').astype(np.float32)
#             enc = (self.latest_depth.encoding or '').lower()
#             if '16uc1' in enc or 'z16' in enc or 'mono16' in enc:
#                 depth_img *= 0.001  # mm -> m
#             dh, dw = depth_img.shape
#             if (self.cam_w and self.cam_h) and (dw != self.cam_w or dh != self.cam_h):
#                 self.get_logger().warn(f"camera_info({self.cam_w}x{self.cam_h}) vs depth({dw}x{dh}) 불일치. 정렬 필요.")
#                 return
#         except Exception:
#             return

#         # detect
#         try:
#             results = self.model.predict(source=cv_img, prompts=self.prompts, verbose=False)
#         except Exception:
#             return
#         if len(results) == 0 or results[0].boxes is None:
#             for label in self.prompts: self._shrink_window(label)
#             return

#         boxes = results[0].boxes
#         confs = boxes.conf.detach().cpu().numpy()
#         if confs.size == 0:
#             for label in self.prompts: self._shrink_window(label)
#             return
#         xyxy_all = boxes.xyxy.detach().cpu().numpy()
#         cls_arr = (boxes.cls.detach().cpu().numpy().astype(int)
#                    if getattr(boxes, 'cls', None) is not None
#                    else np.zeros((xyxy_all.shape[0],), dtype=int))

#         # TF
#         src_frame = self.latest_depth.header.frame_id
#         stamp = self.latest_depth.header.stamp
#         tf = self._lookup_tf(src_frame, stamp)
#         if tf is None:
#             return

#         now = self.get_clock().now()
#         self._prune_tracks(now)

#         fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
#         cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
#         dot = max(2.0 * self.map_res, 0.1)

#         # per box
#         for i in range(xyxy_all.shape[0]):
#             conf = float(confs[i])
#             if conf < self.conf_thresh: continue

#             cls_idx = int(cls_arr[i])
#             label = self.prompts[cls_idx] if (0 <= cls_idx < len(self.prompts)) else self.prompts[0]

#             if not self._hit(label):
#                 continue

#             x1, y1, x2, y2 = xyxy_all[i]
#             u, v = center_from_bbox(x1, y1, x2, y2)

#             # depth (폴백 없이 한 점만)
#             Z = self.depth_at_center(depth_img, u, v)
#             if not (np.isfinite(Z) and 0.05 < Z < 10.0):
#                 continue

#             if self.log_center_uv_only:
#                 self.get_logger().info(f"[center] {label}: u={u:.1f}, v={v:.1f}, Z={Z:.3f} m")

#             if self.publish_center_uv and self.uv_pub is not None:
#                 ps = PointStamped()
#                 ps.header = self.latest_depth.header
#                 ps.point.x, ps.point.y, ps.point.z = float(u), float(v), float(Z)
#                 self.uv_pub.publish(ps)

#             # back-project -> camera -> map
#             X = (u - cx) * Z / fx
#             Y = (v - cy) * Z / fy
#             P_cam = np.array([X, Y, Z], dtype=np.float64)
#             P_map = self.cam_to_map(P_cam, tf)

#             # track match/update
#             j = self._match_track(P_map, label)
#             if j < 0:
#                 # 앵커 락/재결합 검사
#                 k_lock = self._find_anchor_hit(label, P_map)
#                 if k_lock >= 0:
#                     t = self.tracks[k_lock]
#                     t['last_seen'] = now
#                 else:
#                     k_re = self._match_anchor_radius(label, P_map)
#                     if k_re >= 0:
#                         j = k_re
#                     else:
#                         if self._count_label(label) >= self.max_tracks_per_label:
#                             k_any = self._match_track(P_map, label)
#                             if k_any >= 0: j = k_any

#                         if j < 0:
#                             gid = self.next_track_id; self.next_track_id += 1
#                             tname = f"{label}_{gid}"
#                             self.tracks.append({
#                                 'id': gid, 'name': tname, 'label': label,
#                                 'pos': P_map.copy(), 'anchor': P_map.copy(),
#                                 'last_seen': now, 'last_depth_logged': Z
#                             })
#                             self.get_logger().info(f"[{tname}] Depth={Z:.3f} m")
#                             t = self.tracks[-1]
#                             self._publish_marker(t, label, now, dot)
#                             self._publish_radius_markers(t, label, now)
#                             continue

#             # j >= 0 -> update
#             t = self.tracks[j]
#             t['last_seen'] = now
#             if np.linalg.norm(P_map - t['anchor']) <= self.lock_radius:
#                 pass
#             else:
#                 if np.linalg.norm(P_map - t['pos']) > self.update_deadband:
#                     a = self.smooth_alpha
#                     t['pos'] = a * P_map + (1.0 - a) * t['pos']

#             if (t.get('last_depth_logged') is None) or abs(Z - t['last_depth_logged']) >= self.depth_log_delta:
#                 self.get_logger().info(f"[{t['name']}] Depth={Z:.3f} m")
#                 t['last_depth_logged'] = Z

#             self._publish_marker(t, label, now, dot)
#             self._publish_radius_markers(t, label, now)

#     def _publish_marker(self, t, label, stamp, dot):
#         r, g, b = color_for_label(label)
#         marker = Marker()
#         marker.header.frame_id = 'map'
#         marker.header.stamp = stamp.to_msg()
#         marker.ns = f"yolo_{label}"
#         marker.id = int(t['id'])
#         marker.type = Marker.SPHERE
#         marker.action = Marker.ADD
#         marker.pose.position.x = float(t['pos'][0])
#         marker.pose.position.y = float(t['pos'][1])
#         marker.pose.position.z = float(t['pos'][2])
#         marker.scale.x = marker.scale.y = marker.scale.z = float(dot)
#         marker.color.r = float(r); marker.color.g = float(g); marker.color.b = float(b); marker.color.a = 1.0
#         marker.lifetime.sec = int(self.marker_lifetime)
#         marker.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
#         self.marker_pub.publish(marker)


# def main(args=None):
#     rclpy.init(args=args)
#     node = YoloMapMarkerMulti()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#!/usr/bin/env python3
# marker.py
# - conf_thresh=0.15 + (2s, 6-hit) 허들
# - 공간 전역 히트(uv/깊이 근접 3회)로 근거리 안정화
# - 앵커 락 반경 내 재탐지 시 새 마커 생성 금지/기존 트랙 유지
# - "시야-인지 삭제": 앵커가 프레임 안에 보이는 상태로 delete_window_sec 동안 감지 회수가 기준 미만이면 삭제
# - typed_input: 최대 5개 라벨, A/B/C/D/E 색상 고정

#!/usr/bin/env python3
# marker.py
# - conf_thresh=0.15 + (2s, 6-hit) 허들
# - 공간 전역 히트(uv/깊이 근접 3회)로 근거리 안정화
# - 앵커 락 반경 내 재탐지 시 새 마커 생성 금지/기존 트랙 유지
# - "시야-인지 삭제": in-view 상태에서
#       ① focused_delete_window_sec(기본 4s) 동안 히트<2 → 삭제
#       ② delete_window_sec(기본 9s) 동안 히트<2 → 삭제
# - typed_input: 최대 5개 라벨, A/B/C/D/E 색상 고정

import os
from collections import deque, defaultdict
import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLOE

# ------------------------ 유틸 ------------------------
def parse_prompts(text: str, max_n: int = 5):
    if not text:
        return []
    tokens = text.replace(",", " ").split()
    seen = set()
    out = []
    for t in tokens:
        tt = t.strip()
        if not tt:
            continue
        if tt not in seen:
            out.append(tt); seen.add(tt)
        if len(out) >= max_n:
            break
    return out


def bbox_center(x1, y1, x2, y2):
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def fixed_color_for_label(label: str):
    m = {
        'a': (1.0, 0.0, 0.0),  # A red
        'b': (0.0, 0.0, 1.0),  # B blue
        'c': (1.0, 1.0, 1.0),  # C white
        'd': (0.0, 1.0, 0.0),  # D green
        'e': (0.0, 0.0, 0.0),  # E black
    }
    key = (label or "").strip().lower()
    if key in m:
        return m[key]
    palette = [
        (1.0, 0.4, 0.4), (1.0, 0.8, 0.3), (0.4, 1.0, 0.6),
        (0.4, 0.9, 1.0), (0.8, 0.4, 1.0), (1.0, 0.6, 0.9),
    ]
    return palette[abs(hash(label)) % len(palette)]


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


# ------------------------ 노드 ------------------------
class YoloMapMarkerMulti(Node):
    def __init__(self):
        super().__init__("yolo_map_marker_multi")
        self.bridge = CvBridge()

        # --- 파라미터 ---
        self.declare_parameter("map_resolution", 0.05)        # m/pixel
        self.declare_parameter("use_latest_tf", True)         # TF Time(0) 조회
        self.declare_parameter("conf_thresh", 0.15)           # 허들용 신뢰도
        self.declare_parameter("timer_hz", 5.0)               # 기본 5 Hz

        # 라벨 전역 히트(2초, 6회)
        self.declare_parameter("hit_window_sec", 2.0)
        self.declare_parameter("hit_needed", 6)

        # 공간 전역 히트(근거리 안정화)
        self.declare_parameter("birth_window_sec", 2.0)
        self.declare_parameter("birth_needed", 3)
        self.declare_parameter("birth_px_radius", 30.0)
        self.declare_parameter("birth_z_tol", 0.06)

        # 앵커/트랙 관리
        self.declare_parameter("match_radius", 0.20)
        self.declare_parameter("lock_radius", 0.25)
        self.declare_parameter("update_deadband", 0.03)
        self.declare_parameter("smooth_alpha", 0.2)

        # 시야-인지 삭제(샘플 기반 창)
        self.declare_parameter("delete_window_sec", 9.0)      # 일반 창
        self.declare_parameter("delete_min_hits", 2)
        self.declare_parameter("focused_delete_window_sec", 4.0)  # 집중 창
        self.declare_parameter("focused_delete_min_hits", 2)

        # in-view 판정
        self.declare_parameter("view_center_only", False)
        self.declare_parameter("look_center_frac", 0.35)

        # 기타
        self.declare_parameter("anchor_reassoc_radius", 0.45)
        self.declare_parameter("max_tracks_per_label", 12)
        self.declare_parameter("show_radii", True)
        self.declare_parameter("marker_lifetime_sec", 0.0)
        self.declare_parameter("publish_center_uv", False)
        self.declare_parameter("depth_fallback_radius", 1)

        # --- 파라미터 로드 ---
        g = self.get_parameter
        self.map_res                 = float(g("map_resolution").value)
        self.use_latest_tf           = bool(g("use_latest_tf").value)
        self.conf_thresh             = float(g("conf_thresh").value)
        self.timer_hz                = float(g("timer_hz").value)

        self.hit_window_sec          = float(g("hit_window_sec").value)
        self.hit_needed              = int(g("hit_needed").value)

        self.birth_window_sec        = float(g("birth_window_sec").value)
        self.birth_needed            = int(g("birth_needed").value)
        self.birth_px_radius         = float(g("birth_px_radius").value)
        self.birth_z_tol             = float(g("birth_z_tol").value)

        self.match_radius            = float(g("match_radius").value)
        self.lock_radius             = float(g("lock_radius").value)
        self.update_deadband         = float(g("update_deadband").value)
        self.smooth_alpha            = float(g("smooth_alpha").value)

        self.delete_window_sec       = float(g("delete_window_sec").value)
        self.delete_min_hits         = int(g("delete_min_hits").value)
        self.focused_window_sec      = float(g("focused_delete_window_sec").value)
        self.focused_min_hits        = int(g("focused_delete_min_hits").value)

        self.view_center_only        = bool(g("view_center_only").value)
        self.look_center_frac        = float(g("look_center_frac").value)

        self.anchor_reassoc_radius   = float(g("anchor_reassoc_radius").value)
        self.max_tracks_per_label    = int(g("max_tracks_per_label").value)
        self.show_radii              = bool(g("show_radii").value)
        self.marker_lifetime         = float(g("marker_lifetime_sec").value)
        self.publish_center_uv       = bool(g("publish_center_uv").value)
        self.depth_fb_r              = int(g("depth_fallback_radius").value)

        # 샘플 창 길이 (틱 기반)
        self.win_samples_general = max(1, int(round(self.delete_window_sec * self.timer_hz)))
        self.win_samples_focus   = max(1, int(round(self.focused_window_sec * self.timer_hz)))
        self.view_dq_cap         = max(self.win_samples_general, self.win_samples_focus)

        # --- 서브스크립션/퍼블리셔 ---
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_cb, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.rgb_cb, 10)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 10)
        self.create_subscription(String, "/typed_input", self.typed_cb, 10)

        self.marker_pub = self.create_publisher(Marker, "detected_object", 10)
        self.uv_pub     = self.create_publisher(PointStamped, "center_uv", 10) if self.publish_center_uv else None

        # --- TF ---
        self.tfbuf = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuf, self)

        # --- 모델 로드 ---
        model_path = os.path.join(get_package_share_directory("yolo"), "yoloe-11s-seg.pt")
        if not os.path.exists(model_path):
            self.get_logger().error(f"모델 파일 없음: {model_path}")
            raise FileNotFoundError(model_path)
        self.model = YOLOE(model_path, task="segment")
        self.get_logger().info(f"YOLOE 모델 로드: {model_path}")

        # --- 상태 ---
        self.K = None; self.cam_w = None; self.cam_h = None
        self.prompts = []
        self.latest_rgb = None; self.latest_depth = None

        # 전역 히트 카운트
        self.conf_hits = defaultdict(deque)       # label -> deque[stamp]
        self.spatial_hits = defaultdict(deque)    # label -> deque[(stamp,u,v,Z)]

        # 트랙과 라벨별 시퀀스
        # t: {id,name,label,pos,anchor,last_seen,view_samples(deque[bool])}
        self.tracks = []
        self.label_seq = defaultdict(int)         # label -> next id (1..)
        self.next_track_id = 1                    # 내부 유니크 id(마커 id용)

        # 타이머
        self.create_timer(1.0 / self.timer_hz, self.timer_cb)

        self.get_logger().info(
            f"허들(conf>={self.conf_thresh:.2f}): 2s/{self.hit_needed}hit & 공간{self.birth_needed}hit, "
            f"삭제창: focus {self.focused_window_sec:.1f}s<{self.focused_min_hits}hit / "
            f"general {self.delete_window_sec:.1f}s<{self.delete_min_hits}hit"
        )

    # ---------------- 콜백 ----------------
    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.cam_w = int(msg.width); self.cam_h = int(msg.height)
            self.get_logger().info(f"Camera K 로드: {self.cam_w}x{self.cam_h}")

    def rgb_cb(self, msg: Image):   self.latest_rgb = msg
    def depth_cb(self, msg: Image): self.latest_depth = msg

    def typed_cb(self, msg: String):
        prompts = parse_prompts(msg.data, max_n=5)
        self.prompts = prompts
        self.conf_hits.clear()
        self.spatial_hits.clear()
        if self.prompts:
            try:
                pe = self.model.get_text_pe(self.prompts)
                self.model.set_classes(self.prompts, pe)
            except Exception:
                pass
            self.get_logger().info(f"Prompts: {', '.join(self.prompts)}")
            # 프롬프트 변경 시 트랙 리셋
            for t in self.tracks:
                self._delete_markers(t)
            self.tracks.clear()
            self.label_seq.clear()
            self.next_track_id = 1
        else:
            self.get_logger().info("Prompts cleared.")

    # ---------------- TF/투영 ----------------
    def _lookup_tf(self, target_frame: str, src_frame: str, stamp):
        try:
            if self.use_latest_tf:
                return self.tfbuf.lookup_transform(target_frame, src_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2))
            if self.tfbuf.can_transform(target_frame, src_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.5)):
                return self.tfbuf.lookup_transform(target_frame, src_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.2))
            return self.tfbuf.lookup_transform(target_frame, src_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception:
            return None

    def cam_to_map(self, p_cam, tf_cam2map):
        R = quat_to_R(tf_cam2map.transform.rotation)
        o = np.array([tf_cam2map.transform.translation.x,
                      tf_cam2map.transform.translation.y,
                      tf_cam2map.transform.translation.z], dtype=np.float64)
        return R @ p_cam + o

    def map_to_cam(self, p_map, tf_cam2map):
        R = quat_to_R(tf_cam2map.transform.rotation)
        o = np.array([tf_cam2map.transform.translation.x,
                      tf_cam2map.transform.translation.y,
                      tf_cam2map.transform.translation.z], dtype=np.float64)
        return R.T @ (p_map - o)

    def project_map_to_uv(self, p_map, tf_cam2map):
        if self.K is None:
            return None
        p_cam = self.map_to_cam(p_map, tf_cam2map)
        Zc = float(p_cam[2])
        if not np.isfinite(Zc) or Zc <= 0.05:
            return None
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        u = fx * (p_cam[0] / Zc) + cx
        v = fy * (p_cam[1] / Zc) + cy
        return (u, v, Zc)

    # ---------------- 깊이 ----------------
    def depth_at_center(self, depth_img, u, v):
        h, w = depth_img.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return np.nan
        return float(depth_img[vi, ui])

    def depth_at_center_fb(self, depth_img, u, v, r=1):
        Z = self.depth_at_center(depth_img, u, v)
        if np.isfinite(Z) and 0.05 < Z < 10.0:
            return Z
        ui, vi = int(round(u)), int(round(v))
        h, w = depth_img.shape
        vals = []
        for d in range(-r, r+1):
            for (x, y) in ((ui, vi+d), (ui+d, vi)):
                if 0 <= x < w and 0 <= y < h:
                    z = float(depth_img[y, x])
                    if np.isfinite(z) and 0.05 < z < 10.0:
                        vals.append(z)
        return float(np.mean(vals)) if vals else np.nan

    # ---------------- 히트/탄생/매칭 ----------------
    def _shrink_label_hits(self, label):
        # 단순히 창 밖 히트를 제거(시간 계산은 굳이 삭제 로직에 영향 X)
        now = self.get_clock().now()
        dq = self.conf_hits[label]
        while dq and (now - dq[0]).nanoseconds / 1e9 > self.hit_window_sec:
            dq.popleft()

    def _push_label_hit(self, label):
        now = self.get_clock().now()
        dq = self.conf_hits[label]
        dq.append(now)
        self._shrink_label_hits(label)
        return len(dq) >= self.hit_needed

    def _push_spatial_hit(self, label, u, v, Z):
        now = self.get_clock().now()
        dq = self.spatial_hits[label]
        # 창 축소
        while dq and (now - dq[0][0]).nanoseconds / 1e9 > self.birth_window_sec:
            dq.popleft()
        dq.append((now, float(u), float(v), float(Z)))
        # 근접 카운트
        cnt = 0
        for _, uu, vv, zz in dq:
            du = uu - u; dv = vv - v
            if (du*du + dv*dv) ** 0.5 <= self.birth_px_radius and abs(zz - Z) <= self.birth_z_tol:
                cnt += 1
        return cnt >= self.birth_needed

    def _match_track(self, pos_map, label):
        cand = [i for i, t in enumerate(self.tracks) if t["label"] == label]
        if not cand:
            return -1
        dists = [np.linalg.norm(self.tracks[i]["pos"] - pos_map) for i in cand]
        jloc = int(np.argmin(dists))
        return cand[jloc] if dists[jloc] <= self.match_radius else -1

    def _find_anchor_hit(self, label, p_map):
        best_i, best_d = -1, 1e9
        for i, t in enumerate(self.tracks):
            if t["label"] != label:
                continue
            d = float(np.linalg.norm(t["anchor"] - p_map))
            if d < self.lock_radius and d < best_d:
                best_i, best_d = i, d
        return best_i

    # ---------------- in-view & 삭제 ----------------
    def _is_anchor_in_view(self, t, tf_cam2map):
        uvz = self.project_map_to_uv(t["anchor"], tf_cam2map)
        if uvz is None or self.cam_w is None or self.cam_h is None:
            return False
        u, v, Zc = uvz
        if not (0 <= u < self.cam_w and 0 <= v < self.cam_h):
            return False
        if Zc <= 0.05:
            return False
        if not self.view_center_only:
            return True
        cx, cy = self.cam_w * 0.5, self.cam_h * 0.5
        rx, ry = self.cam_w * 0.5 * self.look_center_frac, self.cam_h * 0.5 * self.look_center_frac
        return (abs(u - cx) <= rx) and (abs(v - cy) <= ry)

    def _update_view_samples(self, t, detected_bool):
        dq = t.setdefault("view_samples", deque())
        dq.append(bool(detected_bool))
        while len(dq) > self.view_dq_cap:
            dq.popleft()

    def _sum_last_n(self, dq, n):
        if n <= 0:
            return 0
        if n >= len(dq):
            return sum(1 for v in dq if v)
        s = 0
        for v in list(dq)[-n:]:
            if v: s += 1
        return s

    def _should_delete(self, t):
        dq = t.get("view_samples", deque())
        # 충분한 in-view 샘플이 쌓였을 때만 평가 (틱 기반)
        if len(dq) >= self.win_samples_focus:
            hits_focus = self._sum_last_n(dq, self.win_samples_focus)
            if hits_focus < self.focused_min_hits:
                return True
        if len(dq) >= self.win_samples_general:
            hits_general = self._sum_last_n(dq, self.win_samples_general)
            if hits_general < self.delete_min_hits:
                return True
        return False

    # ---------------- 퍼블리시 ----------------
    def _publish_marker(self, t, label, dot):
        now = self.get_clock().now().to_msg()
        r, g, b = fixed_color_for_label(label)

        mk = Marker()
        mk.header.frame_id = "map"
        mk.header.stamp = now
        mk.ns = f"yolo_{label}"
        mk.id = int(t["id"])
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose.position.x = float(t["pos"][0])
        mk.pose.position.y = float(t["pos"][1])
        mk.pose.position.z = float(t["pos"][2])
        mk.scale.x = mk.scale.y = mk.scale.z = float(dot)
        mk.color.r = float(r); mk.color.g = float(g); mk.color.b = float(b); mk.color.a = 1.0
        mk.lifetime.sec = int(self.marker_lifetime)
        mk.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
        self.marker_pub.publish(mk)

        if self.show_radii:
            mk2 = Marker()
            mk2.header.frame_id = "map"
            mk2.header.stamp = now
            mk2.ns = f"radius_lock_{label}"
            mk2.id = 10000 + int(t["id"])
            mk2.type = Marker.SPHERE
            mk2.action = Marker.ADD
            mk2.pose.position.x = float(t["anchor"][0])
            mk2.pose.position.y = float(t["anchor"][1])
            mk2.pose.position.z = float(t["anchor"][2])
            rlock = float(self.lock_radius)
            mk2.scale.x = mk2.scale.y = mk2.scale.z = 2.0 * rlock
            mk2.color.r = 0.3; mk2.color.g = 1.0; mk2.color.b = 0.3; mk2.color.a = 0.15
            mk2.lifetime.sec = int(self.marker_lifetime)
            mk2.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)
            self.marker_pub.publish(mk2)

    def _delete_markers(self, t):
        now = self.get_clock().now().to_msg()
        mk = Marker()
        mk.header.frame_id = "map"
        mk.header.stamp = now
        mk.ns = f"yolo_{t['label']}"
        mk.id = int(t["id"])
        mk.action = Marker.DELETE
        self.marker_pub.publish(mk)

        mk2 = Marker()
        mk2.header.frame_id = "map"
        mk2.header.stamp = now
        mk2.ns = f"radius_lock_{t['label']}"
        mk2.id = 10000 + int(t["id"])
        mk2.action = Marker.DELETE
        self.marker_pub.publish(mk2)

    # ---------------- 메인 루프 ----------------
    def timer_cb(self):
        if self.K is None or not self.prompts:
            return
        if self.latest_rgb is None or self.latest_depth is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_rgb, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough").astype(np.float32)
            enc = (self.latest_depth.encoding or "").lower()
            if ("16u" in enc) or ("z16" in enc) or ("mono16" in enc):
                depth_img *= 0.001  # mm→m
        except Exception:
            return

        try:
            results = self.model.predict(source=cv_img, prompts=self.prompts, verbose=False)
        except Exception:
            return

        # detection 없는 프레임에서도 삭제 로직은 평가해야 하므로 미리 TF 확보
        src_frame = self.latest_depth.header.frame_id
        stamp = self.latest_depth.header.stamp
        tf_cam2map = self._lookup_tf("map", src_frame, stamp)

        detected_this_frame = {}  # track_id -> bool

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            confs = boxes.conf.detach().cpu().numpy()
            xyxy = boxes.xyxy.detach().cpu().numpy()
            if getattr(boxes, "cls", None) is not None:
                cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
            else:
                cls_arr = np.zeros((xyxy.shape[0],), dtype=int)

            if tf_cam2map is not None:
                fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
                cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
                dot = max(2.0 * self.map_res, 0.10)

                for i in range(xyxy.shape[0]):
                    conf = float(confs[i])
                    if conf < self.conf_thresh:
                        continue

                    cls_idx = int(cls_arr[i])
                    label = self.prompts[cls_idx] if 0 <= cls_idx < len(self.prompts) else self.prompts[0]

                    x1, y1, x2, y2 = xyxy[i]
                    u, v = bbox_center(x1, y1, x2, y2)

                    Z = self.depth_at_center_fb(depth_img, u, v, r=self.depth_fb_r)
                    if not (np.isfinite(Z) and 0.05 < Z < 10.0):
                        continue

                    # 허들(전역+공간)
                    if not (self._push_label_hit(label) and self._push_spatial_hit(label, u, v, Z)):
                        continue

                    # 3D(camera) → map
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    P_cam = np.array([X, Y, Z], dtype=np.float64)
                    P_map = self.cam_to_map(P_cam, tf_cam2map)

                    # 트랙 매칭/생성
                    j = self._match_track(P_map, label)
                    if j < 0:
                        k_lock = self._find_anchor_hit(label, P_map)
                        if k_lock >= 0: j = k_lock

                    if j < 0:
                        # 새 트랙 생성 + 라벨별 일련번호
                        self.label_seq[label] += 1
                        lname = f"{label}_{self.label_seq[label]}"
                        gid = self.next_track_id; self.next_track_id += 1
                        t = {
                            "id": gid, "name": lname, "label": label,
                            "pos": P_map.copy(), "anchor": P_map.copy(),
                            "last_seen": self.get_clock().now(),
                            "view_samples": deque()
                        }
                        self.tracks.append(t)
                        # 허들 최초 통과 로그
                        self.get_logger().info(f"{lname} detected at ({t['pos'][0]:.2f}, {t['pos'][1]:.2f}, {t['pos'][2]:.2f})")
                        # 퍼블리시
                        self._publish_marker(t, label, dot)
                        detected_this_frame[gid] = True
                        continue

                    # 기존 트랙 업데이트
                    t = self.tracks[j]
                    t["last_seen"] = self.get_clock().now()
                    if np.linalg.norm(P_map - t["anchor"]) > self.lock_radius:
                        if np.linalg.norm(P_map - t["pos"]) > self.update_deadband:
                            a = self.smooth_alpha
                            t["pos"] = a * P_map + (1.0 - a) * t["pos"]
                    self._publish_marker(t, label, dot)
                    detected_this_frame[t["id"]] = True

        # --- 시야-인지 삭제 평가(항상 수행) ---
        if tf_cam2map is None:
            return
        keep = []
        for t in self.tracks:
            in_view = self._is_anchor_in_view(t, tf_cam2map)
            if in_view:
                detected = bool(detected_this_frame.get(t["id"], False))
                self._update_view_samples(t, detected)
                if self._should_delete(t):
                    self._delete_markers(t)
                    continue
            keep.append(t)
        self.tracks = keep


def main(args=None):
    rclpy.init(args=args)
    node = YoloMapMarkerMulti()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()