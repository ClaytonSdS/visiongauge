import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import cv2

class ImageDataset(Dataset):
    """
    Dataset class that supports both:
      1. List of image file paths
      2. Tensor of images with shape (batch, C, H, W)

    Args:
        images (list or torch.Tensor): List of paths or tensor (batch, C, H, W)
    """

    def __init__(self, images):
        if isinstance(images, list) and all(isinstance(i, str) for i in images):
            # Case 1: list of file paths
            self.mode = "paths"
            self.images = images
        elif isinstance(images, torch.Tensor):
            # Case 2: preloaded tensor
            if images.ndim != 4:
                raise ValueError(f"Expected tensor of shape (batch, C, H, W), got {images.shape}")
            self.mode = "tensor"
            self.images = images
        else:
            raise ValueError("Input must be a list of paths or a torch tensor (batch, C, H, W)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode == "paths":
            # Load image from file path
            img = cv2.imread(self.images[idx])
            if img is None:
                raise ValueError(f"Could not read image at path: {self.images[idx]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to tensor (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            return img_tensor
        else:
            # Already a tensor (C, H, W)
            return self.images[idx]


class Samples:
    """
    A class to load sample images directly from the Hugging Face UTM_Samples dataset.

    Attributes
    ----------
    images : list of numpy.ndarray
        A list containing the images in RGB format.
    """

    def __init__(self, n_samples=None):
        """
        Initializes the Samples class by loading n_samples images from the
        Hugging Face UTM_Samples dataset.
        """

        dataset = load_dataset("claytonsds/UTM_Samples", split="train")
        self.images = []

        # Use the minimum between requested n_samples and the dataset length
        n_samples = n_samples or len(dataset)
        n_samples = min(n_samples, len(dataset))

        # Load images
        for sample in dataset.select(range(n_samples)):
            img = sample["image"]

            # Convert PIL image to numpy array if necessary
            if hasattr(img, "convert"):
                img = np.array(img)

            # Ensure the image dtype is uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            self.images.append(img)

    def get_images(self):
        """
        Returns all loaded images in RGB format.
        """
        return self.images

    def get_tensors(self):
        """
        Converts all loaded images into PyTorch tensors with shape (C, H, W)
        and stacks them into a single tensor.
        """
        tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in self.images])
        return tensor



s = Samples()