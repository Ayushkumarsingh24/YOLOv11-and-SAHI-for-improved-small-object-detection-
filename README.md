# YOLOv11-and-SAHI (Slicing Aided Hyper Inference)-for-improved-small-object-detection-
# Small object detection is a challenging task in computer vision.
# Performance drops significantly for tiny objects.
# project improves small object detection using YOLO and SAHI
# but Accuracy Decrease While using sahi that a problem here

# Tiny objects occupy very few pixels.
# Standard YOLO struggles with such objects.
# Improvement is needed without retraining the model

# Dataset obtained using Roboflow
# Programming Language: Python Deep Learning Framework: PyTorch Object Detection Model: YOLOv11 Dataset Tool: Roboflow Small Object Enhancement: SAHI (Slicing Aided Hyper Inference) Platform: Google Colab

# 1 Dataset preparation using Roboflow.
# 2 Training YOLOv11 with pre-trained weights.
# 3 Standard YOLOv11 inference.
# 4 Image slicing using SAHI.
# 5 Inference on sliced images.
# 6 Merging predictions using Non-Maximum Suppression (NMS).

# YOLOv11 detected objects effectively.
# Limited performance for very small objects.
# YOLOv11 + SAHI detected more small objects

# Useful in identifying tiny objects that create big problems in real-world scenarios.
