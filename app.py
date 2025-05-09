import streamlit as st
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch import nn

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

MODEL_PATH = "saved_models\efficientnet_asl.pth"
num_classes=29
mean = [0.5187, 0.4988, 0.5147]
std = [0.2017, 0.2310, 0.2390]
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]



model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

#print("model._fc",model._fc)
#print("model._fc.out_features", model._fc.out_features)

#start = time.time()

transform = transforms.Compose([
    transforms.Resize(size = (224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify_sign(image):
    with torch.no_grad():  
        output = model(image)
    predicted_class = output.argmax(dim=1).item()
    return class_names[predicted_class]

st.title("American Sign Language Classification APP")

enable = st.checkbox("Enable camera")
cam = st.camera_input("Show signs to guess your name", disabled=not enable)

if cam:
    st.image(cam)
    image = Image.open(cam)
    image_tensor = transform(image).unsqueeze(0)
    prediction = classify_sign(image_tensor)
    st.write(f"Predicted class: {prediction}")

"""
if enable:
    camera = cv2.VideoCapture(0)
"""

    