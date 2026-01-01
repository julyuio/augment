# Augment for YOLO  
**Simple, readable, and powerful image + bounding‑box augmentation tools**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v5--v11-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A lightweight collection of augmentation utilities designed to help you generate more training data for YOLO models.  
The code is intentionally **simple**, **flat**, and **easy to modify** — no classes, no unnecessary abstractions.

Tested primarily with **YOLOv5 → YOLOv11**, but compatible with any detector using bounding boxes. in the future i will also add polygons for segmentation models. 

Take a look at main.py 

examples.py contains an explanations for all of the function in this library 

---

## 🚀 Features

### 🖼️ Pixel Augmentations
- Adjust contrast & brightness  
- Random HSV shifts  
- Grayscale conversion  
- Noise injection  
- Desaturation  
- Color jitter  
- CLAHE (adaptive histogram equalization)

### 📐 Geometric Augmentations (with BB support)
- Horizontal flip  
- Vertical flip  
- Rotation (with pixel‑accurate BB transformation)  
- Cropping with BB clipping  

### 🧠 Why this repo?
- Clean, readable functions  
- Easy to extend  
- Perfect for dataset bootstrapping  
- Great for debugging and visualization workflows  

---

## 📦 Installation

```bash
git clone https://github.com/julyuio/augment
cd augment
pip install -r requirements.txt