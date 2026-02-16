
# Inference
<a id="top"></a>
* [Inference Example I: List of Image Paths](#inference-example-i-list-of-image-paths)
* [Inference Example II: Tensor Input](#inference-example-ii-tensor-input)
* [Inference Example III: Frame Streaming](#inference-example-iii-frame-streaming)


## Inference Example I: List of Image Paths
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
[↑ Top](#top)

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

[↑ Top](#top)

## Inference Example III: Frame Streaming

To perform inference on a video stream, you must provide an OpenCV capture object as input, like this:

```python
from VisionGauge.models import VisionGauge
import cv2

# Initialize the model
model = VisionGauge()

# Set your camera object
camera = cv2.VideoCapture(0)

# Run inference
model.predict_streaming(
    camera,
    frame_height=1280,
    frame_width=720,
    frame_thickness=4,
    frame_color="#551bb3",
    font_color="#ffffff",
    fontsize=10
)
```
<img src="https://raw.githubusercontent.com/ClaytonSdS/VisionGauge_Files/main/dataset/testing/Oscilation/Filling/streaming.gif" width="800"/>
