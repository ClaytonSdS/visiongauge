# VisionGauge: A computer vision model to detect and read u-tube manometers


This work proposes a computer vision model based on the sequential implementation of machine learning models for the detection and reading of water column gauges, called **VisionGauge**. 

The solution consists of two sequential stages: a **detection model** and a **regression model**. The architecture adopted for the detector is based on the state-of-the-art **YOLOv8** model, while for the regressor, a comparative study was conducted among the **ResNet-18**, **EfficientNetB0**, **MobileNetV3 Small**, and **MobileNetV3 Large** architectures, all adapted for the regression task. 

During training, custom datasets were developed, specific to the training domains of the detector, the regressor, and, finally, for the complete evaluation of the VisionGauge model in both **static** and **streaming** modes. 

The best results composing VisionGauge were obtained with the YOLOv8 model, achieving an **F1-Score of 86.4%**, together with the ResNet-18 architecture, which achieved an **MAE of 1.872**, both evaluated on the test dataset. Additionally, in the streaming mode evaluation, the model achieved an **average oscillation score (Φ) of 70%**.

<img src="https://github.com/ClaytonSdS/VisionGauge_Files/blob/main/steps/visiongauge.png?raw=true" alt="model" width="800"/>


# How to Install

```bash
!pip install visiongauge
```

## Inference Example I: List of image file paths
You can pass a list of image paths like:

```python
from torch.utils.data import DataLoader
from VisionGauge.models import VisionGauge
from VisionGauge.dataset import ImageDataset

# Samples must be a list containing the paths to all images.
# All images must have the same dimensions (height, width, channels).
samples = ["/content/sample3.jpg", "/content/sample4.jpg"]

# Initialize the model
model = VisionGauge()

# Prepare dataset and DataLoader
dataset = ImageDataset(samples)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Run inference
boxes, predictions = model.predict(loader)

# boxes: Returns a torch.Tensor with shape (batch, max_boxes, 4), i.e. boxes[batch, box] = tensor([x1, y1, x2, y2]) coordinates of the current the bounding box.
# predictions: Returns a torch.Tensor with shape (batch, max_boxes, 1) i.e. predictions[batch, box] = tensor([h_p]) fluid height predicted for the current the bounding box.

"""
 Note:
Since some images can generate more than one bounding box, dummy boxes with all-zero coordinates are created,
and the corresponding prediction is set to 0 to maintain consistency and shape during forward propagation.
The number of boxes is determined based on the image with the highest number of predicted bounding boxes, i.e.,
max(bounding_boxes_predicted). This ensures that the predictions tensor has a uniform shape (batch, max_boxes, 1) across the entire batch.
"""
```

## Inference Example II: Tensor input
The model also accepts a tensor in the shape (batch, channels=3, height, width):

```python
import torch
from torch.utils.data import DataLoader
from VisionGauge.models import VisionGauge
from VisionGauge.dataset import ImageDataset, Samples

# Example tensor in the shape (batch_size, 3, height, width)
# e.g., samples = torch.rand((batch_size, 3, 120, 120))
samples = Samples().get_tensors()

# Initialize the model
model = VisionGauge()

# Prepare dataset and DataLoader
dataset = ImageDataset(samples)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Run inference
boxes, predictions = model.predict(loader)

# Plot predictions for image index 0
model.plot_batch(0)
```
<img src="https://github.com/ClaytonSdS/VisionGauge_Files/blob/main/steps/output_example.png?raw=true" alt="model" width="300"/>


