
#%%
from unittest import loader
import numpy as np
from matplotlib import image
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

    def plot_batch(self, batch_number:int=0, figsize:tuple=(6,6)):
        # Check if predictions exist
        if not hasattr(self, 'all_boxes') or not hasattr(self, 'all_predictions'):
            raise ValueError("No predictions available. Please run predict() first.")
        
        # Check if batch_number is valid
        if batch_number >= len(self.all_boxes):
            raise ValueError(f"Invalid batch number. Must be between 0 and {self.all_boxes.shape[0]-1}.")

        # Iterate over the loader saved in self
        for batch_idx, batch_images in enumerate(self.loader):
            if batch_idx != batch_number:
                continue  # skip batches that are not the chosen one

            batch_boxes = self.all_boxes[batch_idx]
            batch_preds = self.all_predictions[batch_idx]

            # Ensure batch_boxes is 2D
            if batch_boxes.dim() == 1:
                batch_boxes = batch_boxes.unsqueeze(0)
                batch_preds = batch_preds.unsqueeze(0)

            # Iterate over all images in the batch
            for img_idx in range(batch_images.shape[0]):
                image = batch_images[img_idx].permute(1, 2, 0).cpu().numpy()  # convert to HWC

                plt.figure(figsize=figsize)
                plt.imshow(image.astype('uint8'))

                # Draw boxes
                for box_idx, box in enumerate(batch_boxes):
                    x1, y1, x2, y2 = box.int().tolist()
                    pred = batch_preds[box_idx].item()

                    # Skip dummy boxes
                    if x1 == y1 == x2 == y2 == 0:
                        continue

                    # Bounding box in color #551bb3
                    rect = plt.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        edgecolor='#551bb3', facecolor='none', linewidth=2
                    )
                    plt.gca().add_patch(rect)

                    # Prediction text in black and bold
                    plt.text(
                        x1, y1-5, f'$h_p$ = {pred:.2f}', 
                        color='black', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
                    )

                plt.axis('off')
                plt.show()
            
            break  # stop after plotting the chosen batch


    def plot_batch_debug(self, batch_number:int=0, figsize:tuple=(6,6)):
        if not hasattr(self, 'all_boxes') or not hasattr(self, 'all_predictions'):
            raise ValueError("No predictions available. Please run predict() first.")
        
        if batch_number >= len(self.loader):
            raise ValueError(f"Invalid batch number. Must be between 0 and {len(self.loader)-1}.")

        for batch_idx, batch_images in enumerate(self.loader):
            if batch_idx != batch_number:
                continue

            batch_boxes = self.all_boxes[batch_idx]
            batch_preds = self.all_predictions[batch_idx]

            print(f"Batch {batch_idx}: batch_images.shape = {batch_images.shape}")
            print(f"Batch {batch_idx}: batch_boxes.shape = {batch_boxes.shape}")
            print(f"Batch {batch_idx}: batch_preds.shape = {batch_preds.shape}")

            for img_idx in range(batch_images.shape[0]):
                image = batch_images[img_idx].permute(1,2,0).cpu().numpy()  # C,H,W -> H,W,C
                print(f"Image {img_idx}: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}, dtype={image.dtype}")
                
                boxes_for_image = batch_boxes[img_idx]
                preds_for_image = batch_preds[img_idx]
                print(f"Image {img_idx}: boxes = {boxes_for_image}")
                print(f"Image {img_idx}: predictions = {preds_for_image}")

                plt.figure(figsize=figsize)
                plt.imshow(np.clip(image, 0, 1))

                for box_idx, box in enumerate(boxes_for_image):
                    x1, y1, x2, y2 = box.int().tolist()
                    pred = preds_for_image[box_idx].item()
                    if x1 == y1 == x2 == y2 == 0:
                        continue

                    rect = plt.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        edgecolor='#551bb3', facecolor='none', linewidth=2
                    )
                    plt.gca().add_patch(rect)
                    plt.text(x1, y1-5, f'$h_p$ = {pred:.2f}', color='black', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

                plt.axis('off')
                plt.show()
            break
