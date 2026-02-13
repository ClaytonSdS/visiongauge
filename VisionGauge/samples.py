import torch
import cv2


import cv2
import torch

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
        self.paths = [rf"samples\\sample{i}.jpg" for i in range(1, 51)]

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
