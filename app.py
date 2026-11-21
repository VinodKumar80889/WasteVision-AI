
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Application Configuration
# -------------------------------

st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide"
)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

MODEL_PATH = "best_resnet18_waste.pth"

device = torch.device("cpu")

# -------------------------------
# 2. Load Trained ResNet18 Model
# -------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


model = load_model()


# -------------------------------
# 3. Grad-CAM Function
# -------------------------------

def generate_gradcam(model, image_tensor, target_class):

    activations = {}
    gradients = {}

    target_layer = model.layer4[-1].conv2

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()

    input_tensor = image_tensor.clone().detach().requires_grad_(True)

    output = model(input_tensor)

    score = output[:, target_class]
    score.backward()

    feature_maps = activations["value"]
    grads = gradients["value"]

    weights = grads.mean(dim=(2, 3), keepdim=True)

    cam = (weights * feature_maps).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    forward_handle.remove()
    backward_handle.remove()

    return cam


# -------------------------------
# 4. Image Preprocessing
# -------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 5. Application Interface
# -------------------------------

st.title("♻️ WasteVision AI")
st.subheader("AI-Based Waste Classification System")

st.write(
    "Upload a waste image and let our deep learning model "
    "identify its category."
)

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# 6. Prediction
# -------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, predicted_index = torch.max(
            probabilities,
            dim=0
        )

    predicted_class = CLASS_NAMES[predicted_index.item()]
    confidence_score = confidence.item() * 100

    with col2:

        st.success("Prediction completed!")

        st.metric(
            "Predicted Waste Category",
            predicted_class.upper()
        )

        st.metric(
            "Model Confidence",
            f"{confidence_score:.2f}%"
        )

    # -------------------------------
    # 7. Class-wise Probabilities
    # -------------------------------

    st.subheader("Class-wise Prediction Probabilities")

    probability_data = {
        CLASS_NAMES[i]: float(probabilities[i].item()) * 100
        for i in range(len(CLASS_NAMES))
    }

    for class_name, probability in probability_data.items():

        st.write(
            f"**{class_name.capitalize()}** — {probability:.2f}%"
        )

        st.progress(int(probability))

    st.caption(
        "Confidence scores are model outputs and are not "
        "guaranteed to be calibrated probabilities."
    )

    # -------------------------------
    # 8. Grad-CAM Visualization
    # -------------------------------

    st.subheader("Grad-CAM Explainability")

    cam = generate_gradcam(
        model,
        image_tensor,
        predicted_index.item()
    )

    import cv2

    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    processed_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])(image)

    original_array = np.array(processed_image)

    overlay = cv2.addWeighted(
        original_array,
        0.6,
        heatmap,
        0.4,
        0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original_array,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col3:
        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )
