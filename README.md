##  VisionExtract: Subject Isolation from Images

A deep learning–based image segmentation system that automatically extracts the main subject from an image and removes the background. The output preserves the subject exactly as in the original image while turning everything else completely black. Built with PyTorch and modern segmentation architectures.

## 📘 About This Project

VisionExtract focuses on semantic segmentation to isolate the primary subject in an image. Given any input image, the model predicts a pixel-wise binary mask that separates the subject from the background, enabling clean subject extraction.

This project is useful for photography automation, digital art, AR/VR pipelines, virtual backgrounds, and content creation workflows.

### 🔍 Features

🧠 Automatic subject detection using deep learning

🎯 Pixel-wise segmentation with binary masks

⬛ Background removal with blacked-out pixels

🖼️ Preserves original subject details and colors

🔁 Supports real-world images beyond the training set

🌐 Optional Streamlit-based inference pipeline

### 🚀 Tech Stack

Framework: PyTorch

Models: UNet (ResNet34), DeepLabV3 (ResNet50)

Data Augmentation: Albumentations

Dataset API: COCO API (PyCOCOtools)

Visualization: Matplotlib

Deployment: Streamlit (optional)

### 🧠 Dataset Info

Source: COCO 2017 Dataset

Images: 118,000+ training images

Validation Set: 5,000 images

Annotations: Pixel-wise segmentation masks

Categories: 80+ object classes

### Why COCO:

High-quality mask annotations

Diverse scenes, lighting, and perspectives

Ideal for real-world generalization

## 🧰 Installation & Usage
### 🔧 Setup Environment

### Clone the repository:

git clone https://github.com/Ilakiya-Emily05/Vision_Ai.git
cd Vision_Ai


### (Optional) Create a virtual environment:

conda create -n visionextract python=3.9
conda activate visionextract


### Install dependencies:

pip install -r requirements.txt


Run inference or training scripts as required.

## ⚙️ Training Pipeline
### 📌 Data Preprocessing

Resize images to 256×256

Normalize pixel values

Convert multi-class masks to binary masks

## Augment data using:

Flips (horizontal & vertical)

Rotation, scaling, shifting

Brightness and contrast adjustments

## 📌 Model Phases

Phase 1: UNet with ResNet34 backbone (baseline)

Phase 2: DeepLabV3 with ResNet50 for improved context & boundaries

## 📌 Optimization

Mixed Precision Training (AMP)

Backbone unfreezing for fine-tuning

Test-Time Augmentation (TTA)

### 📊 Evaluation Metrics

Intersection over Union (IoU)

Dice Coefficient

Pixel-wise Accuracy

Precision & Recall

Both quantitative scores and qualitative before/after visual checks were used.📚 Topics

image-segmentation computer-vision deep-learning pytorch semantic-segmentation coco-dataset streamlit

### 📌 Future Improvements

 Multi-object segmentation

 Transparent background export (PNG)

 Real-time webcam support

 Web deployment (Hugging Face / Streamlit Cloud)

COCO Dataset

Segmentation Models PyTorch

Albumentations

PyTorch

Streamlit
