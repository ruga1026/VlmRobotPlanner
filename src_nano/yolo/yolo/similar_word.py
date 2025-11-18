#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === 기존 코드 (수정 없음) ===
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")

import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import os

# 1) .npy 로딩
names = np.load("src/yolo/lvis_names.npy", allow_pickle=True)  # array of strings
embs_np = np.load("src/yolo/lvis_embeddings.npy")              # shape (N, D), float32
print(names)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 로컬 모델 경로: 환경변수로 덮어쓰기 가능
LOCAL_MODEL_DIR = os.environ.get(
    "SBERT_MODEL_DIR",
    "/home/nano2/ros2_ws/src/yolo/all-MiniLM-L6-v2"  # 필요 시 /tmp/... 로 바꾸세요
)

if not os.path.isdir(LOCAL_MODEL_DIR):
    raise RuntimeError(f"오프라인 모드: 로컬 모델 폴더가 없습니다: {LOCAL_MODEL_DIR}")

device = "cpu"  # torch.distributed 이슈 회피를 위해 CPU 고정
embeddings = torch.from_numpy(embs_np).to(device)      # (N, D)
embeddings = F.normalize(embeddings, p=2, dim=1)       # unit vectors

# 3) 임베딩 모델 초기화 (로컬 모델만 사용)
model = SentenceTransformer(LOCAL_MODEL_DIR, device=device)

def most_similar(query: str, top_n: int = 5):
    """
    query 단어와 의미가 가장 유사한 top_n 단어 리스트를 반환.
    리턴값: List[(단어, 유사도), …]
    """
    # 4) 질의 단어 임베딩
    query_emb = model.encode(query, convert_to_tensor=True)  # (D,)
    query_emb = F.normalize(query_emb, p=2, dim=0)  # unit vector

    # 5) 코사인 유사도
    cos_scores = util.cos_sim(query_emb, embeddings)[0]  # (N,)

    # 6) 상위 N개
    topk = torch.topk(cos_scores, k=top_n)
    indices = topk.indices.tolist()
    scores = topk.values.tolist()

    # 7) (단어, 유사도) 형태로 리턴
    results = []
    for idx, score in zip(indices, scores):
        name = names[idx]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        else:
            name = str(name)
        results.append((name, float(score)))
    return results

# === 여기부터 추가: ROS 2 sub/pub만 추가 ===
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as RosString

class KeywordRankerNode(Node):
    """
    /keywords(String) 구독 → most_similar()로 상위 N 키워드 산출 → /typed_input(String) 퍼블리시
    출력 형식: 'kw1,kw2,kw3,kw4,kw5'
    """
    def __init__(self):
        super().__init__('keyword_ranker')
        # 파라미터: top_n (기본 5) — 구조 변경 없이 옵션만 제공
        self.declare_parameter('top_n', 5)
        self.top_n = int(self.get_parameter('top_n').get_parameter_value().integer_value or 5)

        self.sub = self.create_subscription(RosString, '/keywords', self.on_keywords, 10)
        self.pub = self.create_publisher(RosString, '/typed_input', 10)
        self.get_logger().info(f"Ready: SUB /keywords -> PUB /typed_input (top_n={self.top_n})")

    def on_keywords(self, msg: RosString):
        query = (msg.data or "").strip()
        if not query:
            return

        try:
            ranked = most_similar(query, top_n=self.top_n)
            top_names = [name for name, _ in ranked]
            out_msg = RosString()
            out_msg.data = ",".join(top_names)  # 문자열로 내보냄
            self.pub.publish(out_msg)
            self.get_logger().info(f"IN: '{query}'  OUT: '{out_msg.data}'")
        except Exception as e:
            self.get_logger().error(f"ranking failed: {e}")

def main():
    rclpy.init()
    node = KeywordRankerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()