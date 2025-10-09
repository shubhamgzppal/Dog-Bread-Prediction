import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
import json
import numpy as np
import pandas as pd

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Dog Breed Classifier", layout="centered")
st.title("🐶 Dog Breed Classifier")

# -----------------------------
# Load breed mapping
# -----------------------------
with open("breed_mapping.json", "r") as f:
    breeds = json.load(f)
breed_list = list(breeds.keys())

# -----------------------------
# Model Definition (resnet34)
# -----------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        for param in base_model.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(base_model.children())[:-1]) 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = ResNetClassifier(num_classes=len(breed_list))
    model.load_state_dict(torch.load("resnet_custom_head.pth", map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_idx = torch.topk(probs, 5)

    top_prob = top_prob[0].cpu().numpy()
    top_idx = top_idx[0].cpu().numpy()
    predictions = [(breed_list[idx], prob) for idx, prob in zip(top_idx, top_prob)]
    return predictions

# -----------------------------
# Streamlit UI
# -----------------------------
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        predictions = predict(image)

    # Check confidence threshold
    CONFIDENCE_THRESHOLD = 0.30
    
    # Show top prediction
    best_breed, best_conf = predictions[0]
    if best_conf < CONFIDENCE_THRESHOLD:
        st.warning("🐶 I couldn’t find a dog in this photo. Please try uploading a clearer image of a dog.")
    else:
        st.success(f"🐾 Most likely breed: **{best_breed}** ({best_conf * 100:.2f}%)")

        # Bar chart of top-5 predictions
        st.subheader("Top 5 Predictions")
        breeds_, probs_ = zip(*predictions)
    
        df = pd.DataFrame({
        "Breed": breeds_,
        "Confidence (%)": [round(prob * 100, 2) for prob in probs_]
        })
    
        df["Breed"] = df["Breed"].apply(lambda x: x.replace("_", " ").title())
    
        st.bar_chart(df.set_index("Breed"))

