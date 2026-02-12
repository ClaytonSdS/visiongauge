# %%
import torch
import torch.nn as nn
import numpy as np
import cv2

from torchvision import models
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =========================================================
# REGRESSOR
# =========================================================
class Regressor(nn.Module):
    def __init__(self, image_size: tuple = (120, 120)):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = hf_hub_download(
            repo_id="claytonsds/VisionGauge",
            filename="regressor.pth"
        )

        model_state = torch.load(model_path, map_location="cpu")

        self.backbone = models.resnet18(weights=None)
        out_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(out_feats, 1)

        self.backbone.load_state_dict(model_state, strict=False)

        self.backbone.eval()
        self.to(self.device)

    def forward(self, x):
        return self.backbone(x.to(self.device))


# =========================================================
# DETECTOR
# =========================================================
class Detector:
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="claytonsds/VisionGauge",
            filename="detector.pt"
        )
        self.model = YOLO(model_path)

    def predict_batch(self, imgs):
        return self.model.predict(imgs, verbose=False)


# =========================================================
# VISION GAUGE
# =========================================================
class VisionGauge:
    def __init__(self):
        self.detector = Detector()
        self.regressor = Regressor()

        # Criar transformação UMA vez só
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=120, min_width=120, border_mode=0, value=0),
            A.Resize(120, 120),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    # -----------------------------------------------------
    # Predict principal (otimizado)
    # -----------------------------------------------------
    def predict(self, X):
        """
        X: tensor (B, C, H, W)
        Retorna: (B, max_boxes, 1)
        """

        device = self.regressor.device
        B = X.shape[0]

        # -------------------------------------------------
        # 1️⃣ YOLO em batch
        # -------------------------------------------------
        imgs = [
            X[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            for i in range(B)
        ]

        results = self.detector.predict_batch(imgs)

        raw_boxes = [
            r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            for r in results
        ]

        max_boxes = max(len(b) for b in raw_boxes)
        if max_boxes == 0:
            return torch.zeros(B, 1, 1)

        # -------------------------------------------------
        # 2️⃣ Padding eficiente
        # -------------------------------------------------
        padded_boxes = []

        for boxes in raw_boxes:
            pad = max_boxes - len(boxes)
            if pad > 0:
                boxes = np.vstack([
                    boxes,
                    np.zeros((pad, 4), dtype=np.float32)
                ])
            padded_boxes.append(boxes)

        boxes = torch.from_numpy(
            np.array(padded_boxes, dtype=np.float32)
        )

        # -------------------------------------------------
        # 3️⃣ Extrair crops válidos
        # -------------------------------------------------
        crops = []
        index_map = []

        for b in range(B):
            for i, box in enumerate(boxes[b]):
                x1, y1, x2, y2 = box.int().tolist()

                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue

                crop = X[b, :, y1:y2, x1:x2]

                if crop.shape[1] == 0 or crop.shape[2] == 0:
                    continue

                crop_np = crop.permute(1, 2, 0).cpu().numpy()

                crop_tensor = self.transform(image=crop_np)["image"]
                crops.append(crop_tensor)
                index_map.append((b, i))

        if len(crops) == 0:
            return torch.zeros(B, max_boxes, 1)

        crops = torch.stack(crops).to(device)

        # -------------------------------------------------
        # 4️⃣ Forward único no regressor
        # -------------------------------------------------
        with torch.no_grad():
            outputs = self.regressor(crops).cpu()

        # -------------------------------------------------
        # 5️⃣ Reorganizar saída
        # -------------------------------------------------
        Y = torch.zeros(B, max_boxes, 1)

        for (b, i), out in zip(index_map, outputs):
            Y[b, i] = out

        return Y


# =========================================================
# EXEMPLO DE USO
# =========================================================
def cv2_to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()


image1 = cv2.cvtColor(cv2.imread("image1.jpg"), cv2.COLOR_BGR2RGB)

t1 = cv2_to_tensor(image1)

batch = torch.stack([t1])  # (B, C, H, W)

model = VisionGauge()

output = model.predict(batch)

print("Output shape:", output.shape)
print(output)
