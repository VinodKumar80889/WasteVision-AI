
# ♻️ WasteVision AI
### AI-Based Waste Classification System Using Deep Learning

## 📌 Project Overview

WasteVision AI is a Computer Vision and Deep Learning project that automatically classifies waste images into six different categories using Convolutional Neural Networks (CNN) and Transfer Learning.

The system uses a trained ResNet18 model to predict the waste category from an uploaded image. It also provides prediction confidence, class-wise probabilities, and Grad-CAM visualizations to explain the model's predictions.

## 🎯 Objectives

- Automate waste image classification.
- Classify waste into six categories.
- Compare a custom CNN with ResNet18 transfer learning.
- Improve classification performance using transfer learning.
- Provide visual explanations using Grad-CAM.
- Develop an interactive web application using Streamlit.

## 🗂️ Waste Categories

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep Learning framework |
| Torchvision | Image processing and pretrained models |
| ResNet18 | Transfer Learning architecture |
| CNN | Image feature extraction |
| Streamlit | Web application |
| OpenCV | Image processing and heatmap visualization |
| NumPy | Numerical operations |
| Matplotlib | Visualization |

## 🧠 Model Development

Two deep learning approaches were implemented:

### 1. Custom CNN
- Built a CNN architecture from scratch.
- Used convolution, Batch Normalization, ReLU and MaxPooling.
- Achieved 54.74% test accuracy.

### 2. ResNet18 Transfer Learning
- Used pretrained ImageNet weights.
- Froze most pretrained layers and fine-tuned the final convolutional block.
- Replaced the final fully connected layer for six-class classification.
- Achieved 88.68% test accuracy.

## 📊 Model Performance

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 54.74% |
| ResNet18 | 88.68% |

ResNet18 improved test accuracy by 33.94 percentage points over the custom CNN on the held-out test dataset.

## 🔍 Application Features

- Upload waste images.
- Predict one of six waste categories.
- Display model confidence.
- Display class-wise prediction probabilities.
- Generate Grad-CAM heatmaps.
- Display original image, heatmap and overlay.

## 🔬 Explainable AI – Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize image regions that contribute strongly to the model's prediction.

It helps users inspect the model's visual attention and understand its classification behavior.

## 🏗️ Project Structure

```text
WasteVision_AI/
│
├── app.py
├── best_resnet18_waste.pth
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd WasteVision_AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

### 5. Open in browser

```text
http://localhost:8501
```

## 📈 Future Enhancements

- Multi-object waste detection using YOLO.
- Real-time waste classification using a camera.
- Mobile application integration.
- Waste segregation and recycling recommendations.
- Deployment on a cloud platform.

## 👨‍💻 Project Domain

Artificial Intelligence | Machine Learning | Deep Learning | Computer Vision | Explainable AI

---

**Note:** The reported 88.68% accuracy is measured on this project's held-out test split and may vary on new real-world images.
