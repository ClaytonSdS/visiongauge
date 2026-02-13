# VisionGauge: A computer vision model to detect and read u-tube manometers

This work proposes a computer vision model based on the sequential implementation of machine learning models for the detection and reading of water column gauges, called **VisionGauge**. 

The solution consists of two sequential stages: a **detection model** and a **regression model**. The architecture adopted for the detector is based on the state-of-the-art **YOLOv8** model, while for the regressor, a comparative study was conducted among the **ResNet-18**, **EfficientNetB0**, **MobileNetV3 Small**, and **MobileNetV3 Large** architectures, all adapted for the regression task. 

During training, custom datasets were developed, specific to the training domains of the detector, the regressor, and, finally, for the complete evaluation of the VisionGauge model in both **static** and **streaming** modes. 

The best results composing VisionGauge were obtained with the YOLOv8 model, achieving an **F1-Score of 86.4%**, together with the ResNet-18 architecture, which achieved an **MAE of 1.872**, both evaluated on the test dataset. Additionally, in the streaming mode evaluation, the model achieved an **average oscillation score (Φ) of 70%**.
