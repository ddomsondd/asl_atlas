#import streamlit as st
#import time
#import numpy as np
#import torch
#from PIL import Image
#import torchvision.transforms as transforms
#from efficientnet_pytorch import EfficientNet
#from torch import nn
#
#import mediapipe as mp
#import cv2
#
#pattern_url = "pattern.svg" 
#background = '''
#<style>
#body {
#background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
#background-size: cover;
#}
#</style>
#'''
#
#st.markdown(background, unsafe_allow_html=True)
#
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands()
#mp_draw = mp.solutions.drawing_utils
#
#MODEL_PATH = "saved_models\efficientnet_asl.pth"
#num_classes=29
#mean = [0.5187, 0.4988, 0.5147]
#std = [0.2017, 0.2310, 0.2390]
#class_names = [
#    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
#    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#    'del', 'nothing', 'space'
#]
#
#
#
#model = EfficientNet.from_pretrained('efficientnet-b0')
#model._fc = nn.Linear(model._fc.in_features, num_classes)
#model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
#model.eval()
#
##print("model._fc",model._fc)
##print("model._fc.out_features", model._fc.out_features)
#
##start = time.time()
#
#transform = transforms.Compose([
#    transforms.Resize(size = (224,224)),
#    transforms.ToTensor(),
#    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
#])
#
#def classify_sign(image):
#    with torch.no_grad():  
#        output = model(image)
#    predicted_class = output.argmax(dim=1).item()
#    return class_names[predicted_class]
#
#st.title("American Sign Language Classification APP")
#
#enable = st.checkbox("Enable camera")
#cam = st.camera_input("Show signs to guess your name", disabled=not enable)
#
#if cam:
#    st.image(cam)
#    image = Image.open(cam)
#    image_tensor = transform(image).unsqueeze(0)
#    prediction = classify_sign(image_tensor)
#    st.write(f"Predicted class: {prediction}")
#
#
##if enable:
##    camera = cv2.VideoCapture(0)
#


import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp

num_classes = 29
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

#mniej wydajne: (tensor([0.5187, 0.4988, 0.5147]), tensor([0.2017, 0.2310, 0.2390]))
mean = [0.5173, 0.4976, 0.5131]
std = [0.2046, 0.2340, 0.2418]

#szybsze, ale mniej dokładne: (tensor([0.5162, 0.4961, 0.5119]), tensor([0.2008, 0.2299, 0.2381]))

#WAGI
MODEL_PATH = r"saved_models\EfficientNet\3_epochs\efficientnet_asl.pth"
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

#CAŁY MODEL
#MODEL_PATH = "saved_models\EfficientNet\efficientnet_asl_full.pth"
#model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))
#model.eval()

#for name, param in model.named_parameters():
#    print(name, param.data)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

st.title("American Sign Language Recognition")


if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

camera_button_label = "🟢 Start Camera" if not st.session_state.camera_on else "🔴 Close Camera"
if st.button(camera_button_label):
    st.session_state.camera_on = not st.session_state.camera_on


if st.session_state.camera_on:

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        prediction = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Crop the hand region (naively using bounding box around landmarks)
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
            y_coords = [lm.y for lm in results.multi_hand_landmarks[0].landmark]
            xmin = int(min(x_coords) * w)
            xmax = int(max(x_coords) * w)
            ymin = int(min(y_coords) * h)
            ymax = int(max(y_coords) * h)

            hand_img = frame_rgb[ymin:ymax, xmin:xmax]
            if hand_img.size != 0:
                pil_image = Image.fromarray(hand_img)
                img_tensor = transform(pil_image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    pred_idx = outputs.argmax(1).item()
                    prediction = class_names[pred_idx]

                # Show prediction
                cv2.putText(frame, f'Prediction: {prediction}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
