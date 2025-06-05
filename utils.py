import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
import tensorflow as tf

num_classes = 29
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((64, 64))  # Adjust size as needed
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array


def classify_sign_cnn(image):
    MODEL_PATH = r"saved_models\best_model_cnn.keras"

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    image = Image.open(image).convert("RGB")
    image = image.resize((64, 64))  
    image = np.array(image) / 255.0    
    image = np.expand_dims(image, axis=0)  

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]


    return class_names[predicted_class]


def classify_sign_mobilenet(image):
    MODEL_PATH = r"saved_models\mobilenet_sign_model.keras"

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading MobileNet model: {e}")
        return None

    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))  # MobileNetV2 standard size
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return class_names[predicted_class]

def classify_sign_resnet50_pytorch(image):
    MODEL_PATH = r"saved_models\resnet50_asl_pytorch.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) 

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    predicted_class_idx = output.argmax(dim=1).item()
    return class_names[predicted_class_idx]
    


def classify_sign_efficientnet(image):
    mean = [0.5187, 0.4988, 0.5147]     #[0.5173, 0.4976, 0.5131]
    std = [0.2017, 0.2310, 0.2390]      #[0.2046, 0.2340, 0.2418]

    transform = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    image = Image.open(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    MODEL_PATH = r"saved_models\EfficientNet\3_epochs\efficientnet_asl.pth"
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():  
        output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()
    return class_names[predicted_class]