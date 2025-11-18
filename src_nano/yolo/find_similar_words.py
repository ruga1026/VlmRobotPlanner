import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")

import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F

# 1) .npy 로딩
# names.npy: object array of strings
# embeddings.npy: float32 array of shape (N, D)
names = np.load("lvis_names.npy", allow_pickle=True)  # e.g., array of strings
embs_np = np.load("lvis_embeddings.npy")  # shape (N, D), dtype float32
# print(names)
# 2) Torch tensor 변환 및 장치 고정
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.from_numpy(embs_np).to(device)  # (N, D)
# 필요하면 미리 정규화 (cosine similarity용)
embeddings = F.normalize(embeddings, p=2, dim=1)  # unit vectors

# 3) 임베딩 모델 초기화
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def most_similar(query: str, top_n: int = 5):
    """
    query 단어와 의미가 가장 유사한 top_n 단어 리스트를 반환.
    리턴값: List[(단어, 유사도), …]
    """
    # 4) 질의 단어 임베딩 (tensor, device에 올라감)
    query_emb = model.encode(query, convert_to_tensor=True, device=device)  # (D,)
    query_emb = F.normalize(query_emb, p=2, dim=0)  # unit vector

    # 5) 코사인 유사도: inner product since normalized
    cos_scores = util.cos_sim(query_emb, embeddings)[0]  # (N,)

    # 6) 상위 N개
    topk = torch.topk(cos_scores, k=top_n)
    indices = topk.indices.tolist()
    scores = topk.values.tolist()

    # 7) (단어, 유사도) 형태로 정렬해 리턴 (이미 정렬된 상태)
    results = []
    for idx, score in zip(indices, scores):
        name = names[idx]
        # numpy.str_ or bytes handling
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        else:
            name = str(name)
        results.append((name, float(score)))
    return results

# 사용 예시
if __name__ == "__main__":
    while True:
        word = input("단어를 입력하세요 (종료하려면 'exit' 입력): ")
        if word.lower() == "exit":
            break
        for w, score in most_similar(word, top_n=10):
            print(f"{w:30}  {score:.4f}")