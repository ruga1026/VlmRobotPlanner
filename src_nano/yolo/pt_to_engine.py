# # from ultralytics import YOLO

# # # 1) PyTorch .pt 모델 로드
# # model = YOLO("yoloe-11m-seg-pf.pt")

# # # 2) TensorRT .engine 파일로 export
# # model.export(format="engine", half =True)  # 같은 폴더에 'yolo11n.engine' 생성

# # # 3) 변환된 TensorRT 엔진 로드
# # tensorrt_model = YOLO("yoloe-11m-seg-pf.engine")

# # # 4) 추론 예시 (파일 경로, NumPy 배열, 혹은 ROS 토픽 프레임 등 입력 가능)
# # results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
# # print(results)

# # export_to_engine.py
# from ultralytics import YOLO

# model = YOLO("yoloe-11m-seg-pf.pt") #파일이 디렉토리 없을 경우 자동으로 다운로드

# # TensorRT로 FP16 엔진으로 변환
# model.export(format="engine", half=True)

from ultralytics import YOLOE

model = YOLOE("yoloe-11s-seg.pt", task="segment")
model.export(format="engine")
