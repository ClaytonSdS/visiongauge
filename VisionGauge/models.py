
from unittest import loader
import numpy as np
import torch 
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import albumentations as A
from albumentations.pytorch import ToTensorV2   
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Regressor(nn.Module):
    def __init__(self, image_size: tuple = (120, 120)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Baixar modelo do HuggingFace
        model_path = hf_hub_download(
            repo_id="claytonsds/VisionGauge",
            filename="regressor.pth"
        )

        model_state = torch.load(model_path, map_location="cpu")

        # Backbone ResNet18
        self.backbone = models.resnet18(weights=None)
        out_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(out_feats, 1)

        # Adjust keys from state_dict by removing "backbone."
        new_state_dict = {}
        for k, v in model_state.items():
            if k.startswith("backbone."):
                new_state_dict[k.replace("backbone.", "")] = v
            else:
                new_state_dict[k] = v

        # Carregar pesos
        self.backbone.load_state_dict(new_state_dict, strict=True)

        # Colocar em modo inferência
        self.backbone.eval()
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.backbone(x)

    

class Detector:
    def __init__(self):
        model_state = hf_hub_download(repo_id="claytonsds/VisionGauge", filename="detector.pt")
        self.model = YOLO(model_state)

    def predict(self, img):
        results = self.model.predict(img, verbose=False)
        return results


class VisionGauge:
    """
    VisionGauge is a computer vision model to detect and read u-tube manometers.
    
    Attributes:
        detector (Detector): Object detection model.
        regressor (Regressor): Regression model applied to crops.
    """

    def __init__(self):
        self.detector = Detector()
        self.regressor = Regressor()

    def add_Transformation(self, image, min_size=120):
        """
        Apply preprocessing transformations to an image crop.

        Args:
            image (numpy.ndarray): Image to be transformed.
            min_size (int): Minimum size for padding and resizing.

        Returns:
            torch.Tensor: Transformed image tensor with shape (C, H, W).
        """
        transform = A.Compose([
            A.PadIfNeeded(min_height=min_size, min_width=min_size, border_mode=0),
            A.Resize(120, 120),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return transform(image=image)["image"]

    def get_crops(self, tensor, x1, x2, y1, y2):
        """
        Crop a region from a tensor using bounding box coordinates.

        Args:
            tensor (torch.Tensor): Image tensor with shape (C, H, W).
            x1, x2, y1, y2 (int): Bounding box coordinates.

        Returns:
            torch.Tensor: Cropped tensor with shape (C, y2-y1, x2-x1).
        """
        return tensor[:, y1:y2, x1:x2]

    def predict_bounding_boxes(self, img):
        """
        Run the detector on a single image and return bounding boxes.

        Args:
            img (numpy.ndarray): Input image in HWC format.

        Returns:
            numpy.ndarray: Bounding boxes array of shape (num_boxes, 4) 
                           with coordinates [x1, y1, x2, y2].
        """
        results = self.detector.predict(img)
        boxes = results[0].boxes.xyxy.cpu().numpy() 
        return boxes

    def get_bounding_boxes(self, X):
        import numpy as np
        """
        Detect bounding boxes for a batch of images and pad to match shapes.

        Args:
            X (torch.Tensor): Batch of images with shape (batch, C, H, W).

        Returns:
            torch.Tensor: Padded bounding boxes with shape (batch, max_boxes, 4), 
                          where max_boxes is the max number of boxes detected in the batch.

        Note:
            For images in the batch that have fewer boxes than max_boxes, 
            the remaining boxes are filled with zeros. This ensures that the 
            returned tensor has a uniform shape across the batch.
        """
        
        # Ensure input is a batch tensor
        if X.ndim == 3:
            X = X.unsqueeze(0)

        batch = X.shape[0]
        raw_boxes = []

        for i in range(batch):
            # Convert tensor to numpy image (C,H,W) -> (H,W,C)
            img_np = X[i].permute(1, 2, 0).cpu().numpy()
            boxes = self.predict_bounding_boxes(img_np)  # (num_boxes,4)
            raw_boxes.append(boxes)

        # Find the maximum number of boxes detected in any image in the batch
        max_boxes = max(b.shape[0] for b in raw_boxes)

        # Add padding to ensure all have the same number of boxes (max_boxes)
        padded_boxes = []
        for boxes in raw_boxes:
            pad_size = max_boxes - boxes.shape[0]
            if pad_size > 0:
                padding = np.zeros((pad_size, 4), dtype=np.float32)
                boxes = np.vstack([boxes, padding])
            padded_boxes.append(boxes)

        # Convert to tensor (batch, max_boxes, 4)
        padded_boxes = torch.tensor(np.array(padded_boxes), dtype=torch.float32)
        return padded_boxes


    def not_all_equal2zero(self, *args):
        """
        Check if all provided values are not zero.

        Args:
            *args: Numbers to check.

        Returns:
            bool: True if at least one value is non-zero, False otherwise.
        """
        for arg in args:
            if arg != 0:
                return True
        return False
    
    def annotate_frame(
        self,
        frame,
        boxes,
        predictions,
        frame_color: str = "#551bb3",
        font_color: str = "#ffffff",
        fontsize: int = 10,
        frame_thickness: int = 4,
    ):
        """
        Draw bounding boxes and predictions on a frame.

        Parameters
        ----------
        frame : np.ndarray
            Image in HWC BGR format (OpenCV standard).

        boxes : torch.Tensor or np.ndarray
            Bounding boxes (N, 4) → [x1, y1, x2, y2]

        predictions : torch.Tensor or np.ndarray
            Predictions (N, 1) or (N,)

        Returns
        -------
        np.ndarray
            Annotated frame.
        """

        annotated = frame.copy()

        # Convert HEX → BGR
        rgb = mcolors.to_rgb(frame_color)
        frame_color_tuple = tuple(int(c * 255) for c in rgb[::-1])

        rgb_font = mcolors.to_rgb(font_color)
        font_color_tuple = tuple(int(c * 255) for c in rgb_font[::-1])

        # Convert tensors → numpy if needed
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()

        if hasattr(predictions, "cpu"):
            predictions = predictions.cpu().numpy()

        # Ensure predictions is 1D
        predictions = np.array(predictions).reshape(-1)

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box)

            # Skip dummy boxes
            if x1 == y1 == x2 == y2 == 0:
                continue

            if i >= len(predictions):
                continue

            pred = float(predictions[i])

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                frame_color_tuple,
                frame_thickness
            )

            label = f"h_p = {pred:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = fontsize / 10

            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, 2
            )

            padding = 5 + frame_thickness * 2

            rect_x1 = x1
            rect_x2 = x1 + text_width + padding

            rect_y1 = max(0, y1 - text_height - padding)
            rect_y2 = rect_y1 + text_height + padding

            # Label background
            cv2.rectangle(
                annotated,
                (rect_x1, rect_y1),
                (rect_x2, rect_y2),
                frame_color_tuple,
                -1
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (rect_x1 + padding // 2, rect_y2 - padding // 2),
                font,
                font_scale,
                font_color_tuple,
                2
            )

        return annotated


    def predict(self, X):
        """
        Predict properties for each detected bounding box in a batch of images.

        Steps:
        1. Get bounding boxes for the batch (batch, C, H, W) -> (batch, max_boxes, 4)
        2. Crop each detected box from the image
        3. Apply transformations to each crop (padding and resizing)
        4. Run the regressor on the transformed crops
        5. Return predictions and boxes

        Args:
            X (torch.Tensor or DataLoader): Batch tensor (B, C, H, W) or DataLoader.

        Note:
            For boxes that were not detected in an image (dummy boxes with coordinates all zero),
            the corresponding crop is filled with zeros and the prediction is set to 0.
            This ensures that the predictions tensor has uniform shape (batch, max_boxes, 1) 
            across the batch.

        Returns:
            tuple:
                - torch.Tensor: Bounding boxes (batch, max_boxes, 4)
                - torch.Tensor: Predictions (batch, max_boxes, 1)
        """

        all_boxes = []
        all_predictions = []

        # If input is a DataLoader
        if isinstance(X, torch.utils.data.DataLoader):
            self.loader = X  # save loader for plotting
            iterator = X
        else:
            # If input is a tensor, create a "mini-loader" internally
            if X.ndim == 3:  # single image
                X = X.unsqueeze(0)
            self.loader = torch.utils.data.DataLoader(X, batch_size=16, shuffle=False)
            iterator = self.loader

        # Iterate over batches
        for batch in tqdm(iterator, desc="Processing batches"):
            batches, channels, height, width = batch.shape  # (B, C, H, W)
            boxes = self.get_bounding_boxes(batch)          # (B, max_boxes, 4)
            _, max_boxes, _ = boxes.shape
            Y_Tensor = torch.zeros(batches, max_boxes, 1)

            for b in range(batches):
                for i, box in enumerate(boxes[b]):
                    x1, y1, x2, y2 = box.int().tolist()

                    if self.not_all_equal2zero(x1, y1, x2, y2):
                        predicted_image = self.get_crops(batch[b], x1, x2, y1, y2)
                        h_pred, w_pred = predicted_image.shape[1], predicted_image.shape[2]
                        predicted_image = predicted_image.permute(1,2,0).cpu().numpy()
                        max_size = max(h_pred, w_pred)
                        predicted_image = self.add_Transformation(predicted_image, min_size=max_size).unsqueeze(0)

                        with torch.no_grad():
                            output = self.regressor(predicted_image).squeeze(1)

                        Y_Tensor[b, i] = output
                    else:
                        Y_Tensor[b, i] = 0  # dummy box

            all_boxes.append(boxes)
            all_predictions.append(Y_Tensor)

        # Concatenate all batches
        self.all_boxes = torch.cat(all_boxes, dim=0)
        self.all_predictions = torch.cat(all_predictions, dim=0)

        return self.all_boxes, self.all_predictions
    
    def predict_streaming(
        self,
        camera: cv2.VideoCapture,
        frame_height: int = 1280,
        frame_width: int = 720,
        frame_color: str = "#551bb3",
        font_color: str = "#ffffff",
        fontsize: int = 10,
        frame_thickness: int = 4,
        display: bool = True
    ):
        """
        Run real-time detection and regression on a live video stream.
        Uses annotate_frame() for drawing.
        """

        print("Starting streaming...")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        print("Trying to capture frame...")

        capture_logged = False
        prediction_logged = False

        while True:
            ret, frame = camera.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from camera.")

            if not capture_logged:
                print("Frame capture is working correctly.")
                capture_logged = True

            boxes = self.predict_bounding_boxes(frame)

            if not prediction_logged:
                print("Predicting bounding boxes...")
                prediction_logged = True

            predictions = []

            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)

                if x1 == y1 == x2 == y2 == 0:
                    predictions.append(0.0)
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    predictions.append(0.0)
                    continue

                h, w = crop.shape[:2]
                max_size = max(h, w)

                crop = self.add_Transformation(crop, min_size=max_size).unsqueeze(0)

                with torch.no_grad():
                    pred = self.regressor(crop).item()

                predictions.append(pred)

            predictions = np.array(predictions)

            # 🔥 Usa annotate_frame aqui
            annotated_frame = self.annotate_frame(
                frame,
                boxes,
                predictions,
                frame_color=frame_color,
                font_color=font_color,
                fontsize=fontsize,
                frame_thickness=frame_thickness,
            )

            if display:
                cv2.imshow("VisionGauge Streaming", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

    def plot_batch(self, index: int = 0, figsize: tuple = (6, 6)):
        """
        Plot detected bounding boxes and predictions for a specific image.
        Uses annotate_frame() for drawing.
        """

        if not hasattr(self, "all_boxes") or not hasattr(self, "all_predictions"):
            raise ValueError("No predictions available. Please run predict() first.")

        if index >= len(self.all_boxes):
            raise ValueError(
                f"Invalid image index. Must be between 0 and {self.all_boxes.shape[0] - 1}."
            )

        # Reconstruct all images
        all_images = []
        for batch in self.loader:
            all_images.append(batch)

        all_images = torch.cat(all_images, dim=0)

        if index >= len(all_images):
            raise ValueError(
                f"image_index must be less than {len(all_images) - 1}"
            )

        image_tensor = all_images[index]
        image = image_tensor.permute(1, 2, 0).cpu().numpy()

        # If image is normalized (0–1), convert to uint8
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype("uint8")
        else:
            image = image.astype("uint8")

        # Convert from RGB (matplotlib) to BGR (OpenCV) for annotation
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        boxes = self.all_boxes[index]
        predictions = self.all_predictions[index].squeeze(-1)

        # Use annotate_frame
        annotated_bgr = self.annotate_frame(
            image_bgr,
            boxes,
            predictions,
        )

        # convert back to RGB for plotting
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=figsize)
        plt.imshow(annotated_rgb)
        plt.axis("off")
        plt.show()
