# 🚗 YOLOv11 + SAHI — Improved Small Object Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Detecting the vehicles standard YOLO leaves behind — one slice at a time.**

</div>

---

## 📌 Overview

Small object detection is one of the hardest open problems in computer vision. In a compact traffic scene, cars closest to the camera are large and easy to detect — but vehicles further down the road shrink to a handful of pixels and are frequently **missed entirely** by standard detectors.

This project trains a **YOLOv11** model for vehicle detection and augments it with **SAHI (Slicing Aided Hyper Inference)** — a technique that slices high-resolution images into overlapping tiles, runs inference on each tile, and merges the results — to recover detections on small, distant, and dense objects **without retraining the model.**

> 📽️ Originally presented as an academic project at Rajkiya Engineering College, Kannauj (AKTU), guided by Mr. Shashank Yadav.

---

## ❗ Problem Statement

- High-resolution images get **resized before inference**, and small objects lose critical visual detail in the process.
- Tiny/distant objects occupy **very few pixels**, so standard YOLO struggles to detect them.
- In dense traffic scenes, vehicles further from the camera appear progressively smaller — leading to **missed detections** for anything but the foreground.
- Retraining a bigger/higher-resolution model is expensive; the goal here is to **improve detection at inference time only.**

---

## 🎯 Objective

- Study small object detection techniques in deep learning.
- Train a **YOLOv11** object detection model on a custom vehicle dataset.
- Run standard YOLOv11 inference as a baseline.
- Apply **SAHI** (slicing + merged inference) on top of the trained model.
- Compare **YOLOv11** vs **YOLOv11 + SAHI** on detection count, small-object recall, and accuracy trade-offs.

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| DL Framework | PyTorch |
| Detection Model | YOLOv11 (Ultralytics) |
| Small Object Boost | SAHI (Slicing Aided Hyper Inference) |
| Dataset Annotation/Export | Roboflow |
| Training Platform | Google Colab |

---

## 🗂️ Dataset

- Prepared and exported in **YOLO/COCO format** via **Roboflow**.
- Contains annotated traffic scenes with classes such as **Car, Bus, Lorry/Truck, Motorcycle**.
- Split into training, validation, and test sets.
- Downloaded directly into Google Drive for use inside the Colab training pipeline.

---

## 🔬 Methodology

1. **Dataset preparation** — annotate and export dataset using Roboflow.
2. **Training** — fine-tune YOLOv11 (`yolo11m.pt`) on the custom dataset using the Ultralytics CLI.
3. **Baseline inference** — run standard YOLOv11 inference on test images.
4. **Image slicing** — use SAHI to slice each high-resolution image into overlapping tiles.
5. **Sliced inference** — run YOLOv11 inference independently on every tile.
6. **Merge predictions** — recombine all tile-level detections back into full-image coordinates and apply **Non-Maximum Suppression (NMS)** to remove duplicate boxes.

```bash
# Train YOLOv11 on custom dataset
yolo train model=yolo11m.pt data="data.yaml" epochs=50 imgsz=640

# Validate trained model
yolo val model="runs/detect/train/weights/best.pt" data="data.yaml"
```

```python
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov11",
    model_path="runs/detect/train/weights/best.pt",
    confidence_threshold=0.25,
    image_size=640
)

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

---

## 📊 Results

### Traffic Scene Comparison

| Model | Total Objects Detected | Breakdown |
|---|---|---|
| **YOLOv11** (baseline) | **43** | 26 Cars, 3 Lorries, 14 Motorcycles |
| **YOLOv11 + SAHI** | **59** | 1 Bus, 31 Cars, 8 Lorries, 19 Motorcycles |

**Inference:** SAHI improves detection of small and dense objects — a **+37% increase** in total detected objects in the same scene, largely from vehicles further down the road that the baseline model missed.

### Key Observations
- ✅ YOLOv11 alone detects foreground/large objects effectively.
- ✅ YOLOv11 + SAHI recovers significantly more small, distant, and occluded vehicles.
- ⚠️ **Trade-off:** slicing can introduce a drop in per-detection confidence/accuracy on some tiles (partial objects at slice boundaries, duplicate/fragmented boxes before NMS cleanup) — this is an active limitation, see below.
- ✅ Works entirely at inference time — **no retraining required.**

---

## ⚠️ Limitations & Known Issues

- **Accuracy trade-off with SAHI:** while SAHI increases the *number* of small objects detected, it can reduce per-box confidence/precision in some cases — likely due to objects being split across slice boundaries or duplicate detections not being fully resolved by NMS. Tuning `overlap_height_ratio`, `overlap_width_ratio`, and the NMS IoU threshold is an open area for improvement.
- Slicing increases inference time proportionally to the number of tiles — a real-time/latency trade-off for the accuracy gain.
- Very small/heavily occluded objects at the tile edges can still be missed or double-counted pre-NMS.

---

## 🚀 Future Scope

- Useful for identifying tiny objects that create outsized problems in real-world scenarios.
- Missing small objects can lead to serious **safety and operational risks** (e.g., missed pedestrians/vehicles in traffic monitoring).
- Applicable to **surveillance, traffic monitoring, autonomous systems, drone imagery, and satellite imagery.**
- Can be extended to **medical imaging** for detecting small but clinically significant anomalies.
- Further tuning of slice size/overlap and NMS thresholds to close the accuracy gap introduced by slicing.

---

## ✅ Conclusion

- Small object detection remains a core challenge in computer vision.
- Standard YOLOv11 struggles with tiny/distant objects in dense scenes.
- **SAHI meaningfully improves small-object recall** through slicing-based inference.
- This improvement is achieved **without retraining** the underlying model — making it a practical, low-cost enhancement for any existing YOLO deployment.

---

## 🛠️ Getting Started

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install ultralytics sahi

# Train
yolo train model=yolo11m.pt data="data.yaml" epochs=50 imgsz=640

# Run baseline inference
yolo predict model="runs/detect/train/weights/best.pt" source="path/to/image.jpg"

# Run SAHI-enhanced inference — see notebook for full script
```

The full, reproducible pipeline (dataset setup → training → baseline inference → SAHI-sliced inference → comparison) is available in the notebook: [`YOLO11_and_sahi_for_improved_small_object_detection.ipynb`](./YOLO11_and_sahi_for_improved_small_object_detection.ipynb)

---

## 🙏 Acknowledgements

- **Author:** Ayush Kumar Singh
- **Guide:** Mr. Shashank Yadav
- Rajkiya Engineering College, Kannauj — Dr. A.P.J. Abdul Kalam Technical University

---

<div align="center">

⭐ If you found this useful, consider starring the repo!

</div>
