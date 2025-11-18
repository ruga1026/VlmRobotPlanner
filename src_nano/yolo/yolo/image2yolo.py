import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLOE
import os
from ament_index_python.packages import get_package_share_directory

def parse_prompts(text: str):
    """쉼표/공백 기준으로 프롬프트 분리, 빈 문자열 제거, 중복 제거, 순서 유지"""
    if not text:
        return []
    tokens = text.replace(',', ' ').split()
    seen = set()
    prompts = []
    for t in tokens:
        t = t.strip()
        if t and t not in seen:
            prompts.append(t)
            seen.add(t)
    return prompts


class YoloWithTextInputNode(Node):
    def __init__(self):
        super().__init__('yolo_with_text_node')
        self.bridge = CvBridge()
        self.prompts = []          # 현재 활성 프롬프트 리스트
        self.cls2label = {}        # cls 인덱스 -> 문자열 라벨(우리 프롬프트)
        self.conf_thresh = 0.5     # 오탐 줄이기 위한 기본 임계치
        self.iou_thresh = 0.5

        # depth 최신 프레임 보관
        self.latest_depth_img = None   # np.float32, meters
        self.depth_fb_r = 1            # 십자 평균 fallback 반경

        # 이미지 구독 (RGB)
        self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        # Depth (color 정렬된 depth 사용)
        self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10
        )

        # 텍스트(멀티) 구독: "phone,bottle cup" 등
        self.create_subscription(
            String, '/typed_input', self.text_callback, 10
        )

        # 모델 로드
        model_path = os.path.join(
            get_package_share_directory('yolo'), 'yoloe-11s-seg.pt'
        )
        if not os.path.exists(model_path):
            self.get_logger().error(f"모델 파일 없음: {model_path}")
            raise FileNotFoundError(model_path)

        self.model = YOLOE(model_path, task='segment')
        self.get_logger().info(f"YOLOE 모델 로드 완료: {model_path}")

        cv2.namedWindow("YOLO + Multi Prompt", cv2.WINDOW_NORMAL)

    # ---------------- Depth helpers ----------------
    def depth_callback(self, msg: Image):
        """depth 이미지를 float32 [m]로 보관"""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.array(depth, dtype=np.float32)
            enc = (msg.encoding or '').lower()
            # Z16/16UC1 등은 mm 단위이므로 m로 변환
            if '16u' in enc or 'z16' in enc or 'mono16' in enc:
                depth *= 0.001
            self.latest_depth_img = depth
        except Exception as e:
            self.get_logger().warn(f"depth 변환 오류: {e}")

    def _depth_at_center(self, u: float, v: float):
        """중심 한 점 depth (유효하지 않으면 NaN)"""
        di = self.latest_depth_img
        if di is None:
            return np.nan
        h, w = di.shape
        ui = int(round(u)); vi = int(round(v))
        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            return np.nan
        return float(di[vi, ui])

    def _depth_fallback_cross_mean(self, u: float, v: float, r: int = 1):
        """중심 포함 상하좌우 r 픽셀 평균(유효값만)"""
        di = self.latest_depth_img
        if di is None or r <= 0:
            return np.nan
        h, w = di.shape
        ui = int(round(u)); vi = int(round(v))
        vals = []
        def grab(x, y):
            if 0 <= x < w and 0 <= y < h:
                z = float(di[y, x])
                if np.isfinite(z) and 0.05 < z < 10.0:
                    vals.append(z)
        grab(ui, vi)
        for d in range(1, r + 1):
            grab(ui, vi - d); grab(ui, vi + d)
            grab(ui - d, vi); grab(ui + d, vi)
        return float(np.mean(vals)) if vals else np.nan

    def _depth_at_center_with_fallback(self, u: float, v: float):
        """중심 점 → 유효하면 사용, 아니면 십자 평균"""
        Z = self._depth_at_center(u, v)
        if np.isfinite(Z) and 0.05 < Z < 10.0:
            return Z
        return self._depth_fallback_cross_mean(u, v, self.depth_fb_r)

    # ---------------- Prompts ----------------
    def text_callback(self, msg: String):
        prompts = parse_prompts(msg.data)
        self.prompts = prompts
        if self.prompts:
            text_pe = self.model.get_text_pe(self.prompts)
            self.model.set_classes(self.prompts, text_pe)
            self.cls2label = {i: lbl for i, lbl in enumerate(self.prompts)}
            self.get_logger().info(f"프롬프트 업데이트: {', '.join(self.prompts)}")
        else:
            self.cls2label = {}
            self.get_logger().info("프롬프트 비움")

    # ---------------- Main ----------------
    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if not self.prompts:
                cv2.imshow("YOLO + Multi Prompt", cv_image)
                cv2.waitKey(1)
                return

            results = self.model.predict(
                source=cv_image,
                prompts=self.prompts,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                save=False,
                verbose=False
            )

            annotated = cv_image.copy()
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()

                cls_arr = getattr(boxes, 'cls', None)
                if cls_arr is not None:
                    cls_arr = cls_arr.detach().cpu().numpy()
                else:
                    cls_arr = [0] * len(xyxy)

                for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, cls_arr):
                    if conf < self.conf_thresh:
                        continue

                    # 중심 좌표(실수) 계산
                    x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
                    u_center = (x1f + x2f) / 2.0
                    v_center = (y1f + y2f) / 2.0

                    # Depth(m) 읽기 (fallback 포함)
                    Z = self._depth_at_center_with_fallback(u_center, v_center)

                    # 로그: 중심(u,v) + depth
                    label = self.cls2label.get(int(cls_idx), self.prompts[0])
                    if np.isfinite(Z):
                        self.get_logger().info(
                            f"[center] label={label} u={u_center:.1f} v={v_center:.1f} depth={Z:.3f} m"
                        )
                    else:
                        self.get_logger().info(
                            f"[center] label={label} u={u_center:.1f} v={v_center:.1f} depth=NaN"
                        )

                    # 시각화 (박스/라벨/중심점)
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (255, 0, 0), 2)
                    cv2.putText(
                        annotated, f"{label} {conf:.2f}",
                        (x1i, max(0, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )
                    cv2.circle(annotated, (int(round(u_center)), int(round(v_center))), 3, (255, 0, 0), -1)

            cv2.imshow("YOLO + Multi Prompt", annotated)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"YOLO 추론 중 오류 발생: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloWithTextInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
