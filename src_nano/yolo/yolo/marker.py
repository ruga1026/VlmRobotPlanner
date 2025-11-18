import os, json
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


def parse_prompts(text: str, max_n: int = 5):
    if not text:
        return []
    tokens = text.replace(",", " ").split()
    seen, out = set(), []
    for t in tokens:
        tt = t.strip()
        if not tt: continue
        if tt not in seen:
            out.append(tt); seen.add(tt)
        if len(out) >= max_n: break
    return out

def bbox_center(x1, y1, x2, y2):
    return ((float(x1)+float(x2))*0.5, (float(y1)+float(y2))*0.5)

def fixed_color_for_label(label: str):
    m = {
        'a': (1.0, 0.0, 0.0),  # A red
        'b': (0.0, 0.0, 1.0),  # B blue
        'c': (1.0, 1.0, 1.0),  # C white
        'd': (0.0, 1.0, 0.0),  # D green
        'e': (0.0, 0.0, 0.0),  # E black
    }
    key = (label or "").strip().lower()
    if key in m: return m[key]
    palette = [(1.0,0.4,0.4),(1.0,0.8,0.3),(0.4,1.0,0.6),(0.4,0.9,1.0),(0.8,0.4,1.0),(1.0,0.6,0.9)]
    return palette[abs(hash(label)) % len(palette)]

def quat_to_R(q):
    x,y,z,w = q.x,q.y,q.z,q.w
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=np.float64)


class YoloMapMarkerMulti(Node):
    def __init__(self):
        super().__init__("yolo_map_marker_multi")
        self.bridge = CvBridge()

        # --- Params ---
        self.declare_parameter("map_resolution", 0.05)
        self.declare_parameter("use_latest_tf", True)
        self.declare_parameter("conf_thresh", 0.15)
        self.declare_parameter("timer_hz", 5.0)

        self.declare_parameter("hit_window_sec", 2.0)
        self.declare_parameter("hit_needed", 6)

        self.declare_parameter("birth_window_sec", 2.0)
        self.declare_parameter("birth_needed", 3)
        self.declare_parameter("birth_px_radius", 30.0)
        self.declare_parameter("birth_z_tol", 0.06)

        self.declare_parameter("match_radius", 0.20)
        self.declare_parameter("lock_radius", 0.25)
        self.declare_parameter("update_deadband", 0.03)
        self.declare_parameter("smooth_alpha", 0.2)

        # ⬇️ 집중창만 사용 (일반창 파라미터 제거)
        self.declare_parameter("focused_delete_window_sec", 4.0)
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

        # JSON 저장 경로 (기본: ~/ros2_ws/src/yolo/data/detected_objects.json)
        self.declare_parameter(
            "output_json_path",
            os.path.expanduser("~/ros2_ws/src/yolo/data/detected_objects.json")
        )

        g = self.get_parameter
        self.map_res         = float(g("map_resolution").value)
        self.use_latest_tf   = bool(g("use_latest_tf").value)
        self.conf_thresh     = float(g("conf_thresh").value)
        self.timer_hz        = float(g("timer_hz").value)

        self.hit_window_sec  = float(g("hit_window_sec").value)
        self.hit_needed      = int(g("hit_needed").value)

        self.birth_window_sec= float(g("birth_window_sec").value)
        self.birth_needed    = int(g("birth_needed").value)
        self.birth_px_radius = float(g("birth_px_radius").value)
        self.birth_z_tol     = float(g("birth_z_tol").value)

        self.match_radius    = float(g("match_radius").value)
        self.lock_radius     = float(g("lock_radius").value)
        self.update_deadband = float(g("update_deadband").value)
        self.smooth_alpha    = float(g("smooth_alpha").value)

        # 집중창만
        self.focused_window_sec = float(g("focused_delete_window_sec").value)
        self.focused_min_hits   = int(g("focused_delete_min_hits").value)

        self.view_center_only   = bool(g("view_center_only").value)
        self.look_center_frac   = float(g("look_center_frac").value)

        self.anchor_reassoc_radius = float(g("anchor_reassoc_radius").value)
        self.max_tracks_per_label  = int(g("max_tracks_per_label").value)
        self.show_radii            = bool(g("show_radii").value)
        self.marker_lifetime       = float(g("marker_lifetime_sec").value)
        self.publish_center_uv     = bool(g("publish_center_uv").value)
        self.depth_fb_r            = int(g("depth_fallback_radius").value)

        self.output_json_path      = str(g("output_json_path").value)

        # tick window (집중창만 사용)
        self.win_samples_focus = max(1, int(round(self.focused_window_sec * self.timer_hz)))
        self.view_dq_cap       = self.win_samples_focus

        # Subs/Pubs
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.info_cb, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.rgb_cb, 10)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.depth_cb, 10)
        self.create_subscription(String, "/typed_input", self.typed_cb, 10)

        self.marker_pub = self.create_publisher(Marker, "detected_object", 10)
        self.uv_pub     = self.create_publisher(PointStamped, "center_uv", 10) if self.publish_center_uv else None

        # TF
        self.tfbuf = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuf, self)

        # Model
        model_path = os.path.join(get_package_share_directory("yolo"), "yoloe-11m-seg.pt")
        if not os.path.exists(model_path):
            self.get_logger().error(f"모델 파일 없음: {model_path}")
            raise FileNotFoundError(model_path)
        self.model = YOLOE(model_path, task="segment")
        self.get_logger().info(f"YOLOE 모델 로드: {model_path}")

        # State
        self.K=None; self.cam_w=None; self.cam_h=None
        self.prompts=[]
        self.latest_rgb=None; self.latest_depth=None

        self.conf_hits = defaultdict(deque)      # label -> deque[t]
        self.spatial_hits = defaultdict(deque)   # label -> deque[(t,u,v,Z)]

        self.tracks = []                         # active tracks
        self.label_seq = defaultdict(int)        # per-label sequence (1..)
        self.next_track_id = 1                   # numeric marker id

        # JSON 상태
        self.active_map = {}     # id_str -> {"id":..., "class":..., "map_xy":[x,y]}
        self.active_order = []   # 발견 순서대로 id_str 리스트
        self.json_dirty = True   # 최초 1회 쓰기

        # Timer
        self.create_timer(1.0/self.timer_hz, self.timer_cb)

        self.get_logger().info(
            f"허들(conf>={self.conf_thresh:.2f}): 2s/{self.hit_needed}hit & 공간{self.birth_needed}hit, "
            f"삭제창: focus {self.focused_window_sec:.1f}s<{self.focused_min_hits}hit"
        )

    # ---------- Callbacks ----------
    def info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3,3)
            self.cam_w = int(msg.width); self.cam_h = int(msg.height)
            self.get_logger().info(f"Camera K 로드: {self.cam_w}x{self.cam_h}")

    def rgb_cb(self, msg: Image):   self.latest_rgb = msg
    def depth_cb(self, msg: Image): self.latest_depth = msg

    def typed_cb(self, msg: String):
        prompts = parse_prompts(msg.data, max_n=5)
        self.prompts = prompts
        self.conf_hits.clear(); self.spatial_hits.clear()

        # 프롬프트 변경 → 트랙 및 JSON 리셋
        if self.prompts:
            try:
                pe = self.model.get_text_pe(self.prompts)
                self.model.set_classes(self.prompts, pe)
            except Exception:
                pass
            self.get_logger().info(f"Prompts: {', '.join(self.prompts)}")

        # delete all markers
        for t in self.tracks:
            self._delete_markers(t)
        self.tracks.clear()
        self.label_seq.clear()
        self.next_track_id = 1

        # JSON clear
        self.active_map.clear()
        self.active_order.clear()
        self.json_dirty = True
        self._write_json_if_dirty()

        if not self.prompts:
            self.get_logger().info("Prompts cleared.")

    # ---------- TF & projection ----------
    def _lookup_tf(self, target_frame: str, src_frame: str, stamp):
        try:
            if self.use_latest_tf:
                return self.tfbuf.lookup_transform(target_frame, src_frame, rclpy.time.Time(),
                                                   timeout=rclpy.duration.Duration(seconds=0.2))
            if self.tfbuf.can_transform(target_frame, src_frame, stamp, timeout=rclpy.duration.Duration(seconds=0.5)):
                return self.tfbuf.lookup_transform(target_frame, src_frame, stamp,
                                                   timeout=rclpy.duration.Duration(seconds=0.2))
            return self.tfbuf.lookup_transform(target_frame, src_frame, rclpy.time.Time(),
                                               timeout=rclpy.duration.Duration(seconds=0.2))
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
        if self.K is None: return None
        p_cam = self.map_to_cam(p_map, tf_cam2map)
        Zc = float(p_cam[2])
        if not np.isfinite(Zc) or Zc <= 0.05: return None
        fx,fy = float(self.K[0,0]), float(self.K[1,1])
        cx,cy = float(self.K[0,2]), float(self.K[1,2])
        u = fx*(p_cam[0]/Zc) + cx
        v = fy*(p_cam[1]/Zc) + cy
        return (u,v,Zc)

    # ---------- depth ----------
    def depth_at_center(self, depth_img, u, v):
        h,w = depth_img.shape
        ui,vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h): return np.nan
        return float(depth_img[vi,ui])

    def depth_at_center_fb(self, depth_img, u, v, r=1):
        Z = self.depth_at_center(depth_img,u,v)
        if np.isfinite(Z) and 0.05 < Z < 10.0: return Z
        ui,vi = int(round(u)), int(round(v))
        h,w = depth_img.shape
        vals=[]
        for d in range(-r,r+1):
            for (x,y) in ((ui,vi+d),(ui+d,vi)):
                if 0<=x<w and 0<=y<h:
                    z = float(depth_img[y,x])
                    if np.isfinite(z) and 0.05<z<10.0: vals.append(z)
        return float(np.mean(vals)) if vals else np.nan

    # ---------- hits / matching ----------
    def _shrink_label_hits(self, label):
        now = self.get_clock().now()
        dq = self.conf_hits[label]
        while dq and (now - dq[0]).nanoseconds/1e9 > self.hit_window_sec:
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
        while dq and (now - dq[0][0]).nanoseconds/1e9 > self.birth_window_sec:
            dq.popleft()
        dq.append((now,float(u),float(v),float(Z)))
        cnt=0
        for _,uu,vv,zz in dq:
            du=uu-u; dv=vv-v
            if (du*du+dv*dv)**0.5 <= self.birth_px_radius and abs(zz - Z) <= self.birth_z_tol:
                cnt+=1
        return cnt >= self.birth_needed

    def _match_track(self, pos_map, label):
        cand=[i for i,t in enumerate(self.tracks) if t["label"]==label]
        if not cand: return -1
        dists=[np.linalg.norm(self.tracks[i]["pos"]-pos_map) for i in cand]
        jloc=int(np.argmin(dists))
        return cand[jloc] if dists[jloc] <= self.match_radius else -1

    def _find_anchor_hit(self, label, p_map):
        best_i, best_d = -1, 1e9
        for i,t in enumerate(self.tracks):
            if t["label"]!=label: continue
            d=float(np.linalg.norm(t["anchor"]-p_map))
            if d < self.lock_radius and d < best_d:
                best_i, best_d = i, d
        return best_i

    # ---------- in-view / delete ----------
    def _is_anchor_in_view(self, t, tf_cam2map):
        uvz = self.project_map_to_uv(t["anchor"], tf_cam2map)
        if uvz is None or self.cam_w is None or self.cam_h is None: return False
        u,v,Zc = uvz
        if not (0 <= u < self.cam_w and 0 <= v < self.cam_h): return False
        if Zc <= 0.05: return False
        if not self.view_center_only: return True
        cx,cy = self.cam_w*0.5, self.cam_h*0.5
        rx,ry = self.cam_w*0.5*self.look_center_frac, self.cam_h*0.5*self.look_center_frac
        return (abs(u-cx)<=rx) and (abs(v-cy)<=ry)

    def _update_view_samples(self, t, detected_bool):
        dq = t.setdefault("view_samples", deque())
        dq.append(bool(detected_bool))
        while len(dq) > self.view_dq_cap:
            dq.popleft()

    def _sum_last_n(self, dq, n):
        if n <= 0: return 0
        if n >= len(dq): return sum(1 for v in dq if v)
        s=0
        for v in list(dq)[-n:]:
            if v: s+=1
        return s

    def _should_delete(self, t):
        dq = t.get("view_samples", deque())
        if len(dq) >= self.win_samples_focus:
            if self._sum_last_n(dq, self.win_samples_focus) < self.focused_min_hits:
                return True
        return False

    # ---------- JSON helpers ----------
    def _round2(self, x):  # 보기 좋게 소수 2자리
        return float(np.round(x, 2))

    def _json_set(self, id_str, label, x, y):
        """추가/갱신, 발견 순서 유지."""
        item = {"id": id_str, "class": label, "map_xy": [self._round2(x), self._round2(y)]}
        if id_str not in self.active_map:
            self.active_map[id_str] = item
            self.active_order.append(id_str)
        else:
            prev = self.active_map[id_str]
            if prev["map_xy"] != item["map_xy"]:
                self.active_map[id_str] = item
        self.json_dirty = True

    def _json_del(self, id_str):
        if id_str in self.active_map:
            del self.active_map[id_str]
            try:
                self.active_order.remove(id_str)
            except ValueError:
                pass
            self.json_dirty = True

    def _write_json_if_dirty(self):
        if not self.json_dirty: return
        try:
            out_path = os.path.expanduser(self.output_json_path)
            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            objects = [self.active_map[i] for i in self.active_order if i in self.active_map]
            data = [{"objects": objects}]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            self.json_dirty = False
        except Exception as e:
            self.get_logger().warn(f"JSON 쓰기 실패: {e}")

    # ---------- markers ----------
    def _publish_marker(self, t, label, dot):
        now = self.get_clock().now().to_msg()
        r,g,b = fixed_color_for_label(label)
        mk = Marker()
        mk.header.frame_id = "map"; mk.header.stamp = now
        mk.ns = f"yolo_{label}"; mk.id = int(t["id"])
        mk.type = Marker.SPHERE; mk.action = Marker.ADD
        mk.pose.position.x = float(t["pos"][0])
        mk.pose.position.y = float(t["pos"][1])
        mk.pose.position.z = float(t["pos"][2])
        mk.scale.x = mk.scale.y = mk.scale.z = float(dot)
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = float(r), float(g), float(b), 1.0
        mk.lifetime.sec = int(self.marker_lifetime)
        mk.lifetime.nanosec = int((self.marker_lifetime-int(self.marker_lifetime))*1e9)
        self.marker_pub.publish(mk)

        if self.show_radii:
            mk2 = Marker()
            mk2.header.frame_id = "map"; mk2.header.stamp = now
            mk2.ns = f"radius_lock_{label}"; mk2.id = 10000 + int(t["id"])
            mk2.type = Marker.SPHERE; mk2.action = Marker.ADD
            mk2.pose.position.x = float(t["anchor"][0])
            mk2.pose.position.y = float(t["anchor"][1])
            mk2.pose.position.z = float(t["anchor"][2])
            rlock = float(self.lock_radius)
            mk2.scale.x = mk2.scale.y = mk2.scale.z = 2.0 * rlock
            mk2.color.r, mk2.color.g, mk2.color.b, mk2.color.a = 0.3, 1.0, 0.3, 0.15
            mk2.lifetime.sec = int(self.marker_lifetime)
            mk2.lifetime.nanosec = int((self.marker_lifetime-int(self.marker_lifetime))*1e9)
            self.marker_pub.publish(mk2)

    def _delete_markers(self, t):
        now = self.get_clock().now().to_msg()
        mk = Marker()
        mk.header.frame_id = "map"; mk.header.stamp = now
        mk.ns = f"yolo_{t['label']}"; mk.id = int(t["id"])
        mk.action = Marker.DELETE
        self.marker_pub.publish(mk)
        mk2 = Marker()
        mk2.header.frame_id = "map"; mk2.header.stamp = now
        mk2.ns = f"radius_lock_{t['label']}"; mk2.id = 10000 + int(t["id"])
        mk2.action = Marker.DELETE
        self.marker_pub.publish(mk2)

    # ---------- main loop ----------
    def timer_cb(self):
        # 매 틱 말미에 json을 기록
        def finish():
            self._write_json_if_dirty()

        if self.K is None or not self.prompts:
            return finish()
        if self.latest_rgb is None or self.latest_depth is None:
            return finish()

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_rgb, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough").astype(np.float32)
            enc = (self.latest_depth.encoding or "").lower()
            if ("16u" in enc) or ("z16" in enc) or ("mono16" in enc): depth_img *= 0.001
        except Exception:
            return finish()

        try:
            results = self.model.predict(source=cv_img, prompts=self.prompts, verbose=False)
        except Exception:
            return finish()

        src_frame = self.latest_depth.header.frame_id
        stamp = self.latest_depth.header.stamp
        tf_cam2map = self._lookup_tf("map", src_frame, stamp)

        detected_this_frame = {}

        if results and results[0].boxes is not None and tf_cam2map is not None:
            boxes = results[0].boxes
            confs = boxes.conf.detach().cpu().numpy()
            xyxy  = boxes.xyxy.detach().cpu().numpy()
            if getattr(boxes, "cls", None) is not None:
                cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
            else:
                cls_arr = np.zeros((xyxy.shape[0],), dtype=int)

            fx,fy = float(self.K[0,0]), float(self.K[1,1])
            cx,cy = float(self.K[0,2]), float(self.K[1,2])
            dot = max(2.0*self.map_res, 0.10)

            for i in range(xyxy.shape[0]):
                conf = float(confs[i])
                if conf < self.conf_thresh: continue
                cls_idx = int(cls_arr[i])
                label = self.prompts[cls_idx] if 0<=cls_idx < len(self.prompts) else self.prompts[0]
                x1,y1,x2,y2 = xyxy[i]
                u,v = bbox_center(x1,y1,x2,y2)

                Z = self.depth_at_center_fb(depth_img, u, v, r=self.depth_fb_r)
                if not (np.isfinite(Z) and 0.05 < Z < 10.0): continue
                if not (self._push_label_hit(label) and self._push_spatial_hit(label, u, v, Z)): continue

                X = (u-cx)*Z/fx; Y=(v-cy)*Z/fy
                P_map = self.cam_to_map(np.array([X,Y,Z],dtype=np.float64), tf_cam2map)

                j = self._match_track(P_map, label)
                if j < 0:
                    k_lock = self._find_anchor_hit(label, P_map)
                    if k_lock >= 0: j = k_lock

                if j < 0:
                    # 새 트랙 생성 (로그/JSON 표기 모두 label#seq)
                    self.label_seq[label] += 1
                    seq = self.label_seq[label]
                    lid_hash = f"{label}#{seq}"
                    gid = self.next_track_id; self.next_track_id += 1
                    t = {
                        "id": gid, "name": lid_hash, "json_id": lid_hash,
                        "label": label,
                        "pos": P_map.copy(), "anchor": P_map.copy(),
                        "last_seen": self.get_clock().now(),
                        "view_samples": deque()
                    }
                    self.tracks.append(t)
                    self.get_logger().info(f"{lid_hash} detected at ({t['pos'][0]:.2f}, {t['pos'][1]:.2f}, {t['pos'][2]:.2f})")
                    self._publish_marker(t, label, dot)
                    detected_this_frame[gid] = True
                    # JSON add
                    self._json_set(t["json_id"], label, t["pos"][0], t["pos"][1])
                    continue

                # 기존 트랙 업데이트
                t = self.tracks[j]
                t["last_seen"] = self.get_clock().now()
                if np.linalg.norm(P_map - t["anchor"]) > self.lock_radius:
                    if np.linalg.norm(P_map - t["pos"]) > self.update_deadband:
                        a = self.smooth_alpha
                        t["pos"] = a * P_map + (1.0 - a) * t["pos"]
                        # JSON update
                        self._json_set(t["json_id"], t["label"], t["pos"][0], t["pos"][1])
                detected_this_frame[t["id"]] = True
                self._publish_marker(t, t["label"], dot)

        # in-view 삭제 및 JSON 제거 (집중창만)
        if tf_cam2map is not None:
            keep=[]
            for t in self.tracks:
                in_view = self._is_anchor_in_view(t, tf_cam2map)
                if in_view:
                    detected = bool(detected_this_frame.get(t["id"], False))
                    self._update_view_samples(t, detected)
                    if self._should_delete(t):
                        self._delete_markers(t)
                        self._json_del(t["json_id"])
                        continue
                keep.append(t)
            self.tracks = keep

        # 매 틱 마지막에 JSON 파일 갱신
        self._write_json_if_dirty()


def main(args=None):
    rclpy.init(args=args)
    node = YoloMapMarkerMulti()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()