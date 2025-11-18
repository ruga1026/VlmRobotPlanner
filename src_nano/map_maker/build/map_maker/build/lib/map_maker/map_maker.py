#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Humble: Subscribe to /map and TF, overlay world origin + robot pose, save PNG periodically.

pip install opencv-python numpy
"""

import math
from typing import Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped

import tf2_ros


def yaw_from_quat(q):
    # quaternion -> yaw (Z)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_image(xw, yw, origin_xy_yaw, resolution, img_height):
    """
    Convert world (xw,yw) [m] to image pixel (col,row) for a ROS map:
      - origin [x0,y0,yaw] is world pose of cell (0,0) (lower-left).
      - meters->cells u,v via inverse rotation/translation (map frame).
      - image rows are top->down: row = (H-1) - v
    """
    x0, y0, yaw = origin_xy_yaw
    dx = xw - x0
    dy = yw - y0
    c = math.cos(yaw)
    s = math.sin(yaw)
    u_m = c * dx + s * dy
    v_m = -s * dx + c * dy
    u = u_m / resolution
    v = v_m / resolution
    col = u
    row = (img_height - 1) - v
    return float(col), float(row)


def occ_grid_to_gray_img(grid: OccupancyGrid) -> np.ndarray:
    """
    Convert OccupancyGrid to a displayable grayscale image (uint8, top-left origin).
    Convention: -1 (unknown) -> 205, 0 (free) -> 254..255, 100 (occupied) -> 0.
    """
    w = grid.info.width
    h = grid.info.height
    data = np.asarray(grid.data, dtype=np.int16).reshape((h, w))

    img = np.empty((h, w), dtype=np.uint8)
    unknown = data < 0
    known = ~unknown
    img[unknown] = 205
    # Scale occupied 0..100 to 255..0 (occupied->dark)
    clipped = np.clip(data[known], 0, 100)
    img[known] = (255 - (clipped * 255 // 100)).astype(np.uint8)

    # Flip vertically to match image top-left origin
    img = np.flipud(img)
    return img


class MapAnnotatorNode(Node):
    def __init__(self):
        super().__init__("map_annotator")

        # Parameters
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("save_path", "annotated_map.png")
        self.declare_parameter("save_period_sec", 1.0)
        self.declare_parameter("draw_origin", True)
        self.declare_parameter("arrow_len_m", 0.5)  # robot heading arrow length
        self.declare_parameter("text_scale", 0.5)

        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.save_path = self.get_parameter("save_path").get_parameter_value().string_value
        self.save_period = float(self.get_parameter("save_period_sec").value)
        self.draw_origin = bool(self.get_parameter("draw_origin").value)
        self.arrow_len_m = float(self.get_parameter("arrow_len_m").value)
        self.text_scale = float(self.get_parameter("text_scale").value)

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Map subscription
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, 10
        )

        # Timer for periodic save
        self.timer = self.create_timer(self.save_period, self.on_timer)

        self.get_logger().info(
            f"MapAnnotator running. map_topic={self.map_topic}, map_frame={self.map_frame}, "
            f"base_frame={self.base_frame}, save_path={self.save_path}, period={self.save_period}s"
        )

    def on_map(self, msg: OccupancyGrid):
        self.map_msg = msg

    def get_robot_pose(self):
        """
        Latest transform map->base_link. Returns (x,y,yaw) in map frame, or None on failure.
        """
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

    def annotate(self, gray_img: np.ndarray, grid: OccupancyGrid) -> np.ndarray:
        """
        Draw origin and robot on a BGR surface based on the given gray image and metadata.
        """
        h, w = gray_img.shape[:2]
        bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Extract origin pose from grid.info.origin
        origin_pose = grid.info.origin
        x0 = float(origin_pose.position.x)
        y0 = float(origin_pose.position.y)
        yaw0 = yaw_from_quat(origin_pose.orientation)
        origin = (x0, y0, yaw0)
        res = float(grid.info.resolution)

        # Draw world origin (0,0) if requested
        if self.draw_origin:
            col_f, row_f = world_to_image(0.0, 0.0, origin, res, h)
            ci, ri = int(round(col_f)), int(round(row_f))
            if 0 <= ci < w and 0 <= ri < h:
                cv2.arrowedLine(bgr, (max(0, ci - 40), ri), (ci, ri), (0, 0, 255), 2, tipLength=0.25)
                cv2.circle(bgr, (ci, ri), 3, (0, 0, 255), -1)
                cv2.putText(bgr, "world (0,0)", (min(ci + 6, w - 1), max(ri - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw robot (latest TF)
        rp = self.get_robot_pose()
        if rp is not None:
            xw, yw, yaw_r = rp
            col_r, row_r = world_to_image(xw, yw, origin, res, h)
            cri, rri = int(round(col_r)), int(round(row_r))

            # heading arrow by projecting a point ahead in world then mapping
            ahead = self.arrow_len_m
            xh = xw + ahead * math.cos(yaw_r)
            yh = yw + ahead * math.sin(yaw_r)
            col_h, row_h = world_to_image(xh, yh, origin, res, h)

            if 0 <= cri < w and 0 <= rri < h:
                cv2.circle(bgr, (cri, rri), 4, (0, 255, 0), -1)
                cv2.arrowedLine(bgr, (cri, rri), (int(round(col_h)), int(round(row_h))),
                                (0, 255, 0), 2, tipLength=0.25)
                label = f"robot ({xw:.2f},{yw:.2f})"
                cv2.putText(bgr, label, (min(cri + 6, w - 1), max(rri - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                self.get_logger().warn(f"Robot projects outside image: (col={col_r:.1f}, row={row_r:.1f})")
        else:
            # no TF available; just leave the map as-is
            pass

        return bgr

    def on_timer(self):
        if self.map_msg is None:
            return  # no map yet

        try:
            gray = occ_grid_to_gray_img(self.map_msg)
            annotated = self.annotate(gray, self.map_msg)

            # Save both annotated and plain for convenience
            cv2.imwrite(self.save_path, annotated)
            cv2.imwrite(self._paired_path(self.save_path, suffix="_plain", ext=".png"), gray)
        except Exception as e:
            self.get_logger().error(f"Failed to render/save: {e}")

    @staticmethod
    def _paired_path(path: str, suffix="_plain", ext=".png") -> str:
        import os
        base, _ = os.path.splitext(path)
        return base + suffix + ext


def main():
    rclpy.init()
    node = MapAnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
