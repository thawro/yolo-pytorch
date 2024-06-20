import onnx
import torch
from onnxsim import simplify

from src.detection.architectures.yolo_v10 import (
    YOLOv10,
    YOLOv10b,
    YOLOv10l,
    YOLOv10m,
    YOLOv10n,
    YOLOv10s,
    YOLOv10x,
)
from src.utils.torch_utils import model_info

Classes = [YOLOv10n, YOLOv10s, YOLOv10m, YOLOv10b, YOLOv10l, YOLOv10x]
versions = ["N", "S", "M", "B", "L", "X"]

x = torch.randn(1, 3, 640, 640)

model = YOLOv10s(num_classes=80)
model.fuse()
f = "model.onnx"
torch.onnx.export(model, args=x, f=f)
model_onnx = onnx.load(f)
model_simp, check = simplify(model_onnx)
onnx.save(model_simp, f)

for version in versions:
    model = YOLOv10(version, num_classes=80)
    model.fuse()
    model_info(model, imgsz=640)
    print("=" * 100)

# exit()
# for Class in Classes:
#     model = Class(num_classes=80)
#     model.fuse()
#     model_info(model, imgsz=640)
#     print("=" * 100)

# model_2 = YOLOv10("yolov10s.pt")
# print(model)
# print(model_2)

# model_2.model.model[-1].export = True
# model_2.model.model[-1].format = "onnx"
# del model_2.model.model[-1].cv2
# del model_2.model.model[-1].cv3
# model_2 = model_2.model


preds = model(x)

# model_info(model_2, imgsz=640)

# # model_2.training = True
# preds_2 = model_2(x)["one2many"]


for i in range(3):
    print(preds[0][i].shape)
