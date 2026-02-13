from torch.utils.data import Dataset
import torch
import cv2
import os

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
    A class to load sample images from dataset.

    Attributes
    ----------
    paths : list of str
        A list containing the file paths for the sample images.
    """

    def __init__(self):
        """
        Initializes the Samples class by creating a list of file paths for
        50 sample images located in the 'samples' folder.
        """
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        samples_dir = os.path.join(root_dir, "samples")

        # Cria caminhos completos para os 50 arquivos sample1.jpg até sample50.jpg
        self.paths = [os.path.join(samples_dir, f"sample{i}.jpg") for i in range(1, 51)]


    def get_images(self):
        """
        Loads all sample images from the stored file paths and converts them
        from BGR to RGB format.

        Returns
        -------
        list of numpy.ndarray
            A list of images in RGB format.
        """
        return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in self.paths]
    
    def get_tensors(self):
        """
        Converts all loaded images into PyTorch tensors with shape (C, H, W)
        and stacks them into a single tensor.

        Returns
        -------
        torch.Tensor
            A tensor containing all images stacked along the first dimension.
            Shape: (number_of_images, channels, height, width)
        """
        images = self.get_images()
        tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in images])
        return tensor
