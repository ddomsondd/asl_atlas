import torch
from torchvision import transforms, models
from PIL import Image
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === ŚCIEŻKI ===
test_root = r"C:\Users\azydo\OneDrive(1)\Pulpit\Nowy folder\zdjecia do sprawdzenia 2" #tu dodać ścieżkę do folderu ze zdjęciami
model_path = "models_resnet50/resnet50_asl_aug1_2_dataset_extra.pth" #tu dodać swoją ścieżkę do modelu
train_class_folder = "dataset/asl_alphabet_train"

# === WCZYTANIE MODELU ===
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 29)
model.load_state_dict(torch.load(model_path))
model.eval()

# === TRANSFORMACJA ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === KLASY ===
class_names = sorted(os.listdir(train_class_folder))
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

y_true = []
y_pred = []

# === PREDYKCJE ===
for true_class in class_names:
    class_folder = os.path.join(test_root, true_class)
    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):
        file_path = os.path.join(class_folder, filename)
        try:
            img = Image.open(file_path).convert("RGB")
        except:
            print(f"Błąd odczytu: {file_path}")
            continue

        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        y_true.append(class_to_idx[true_class])
        y_pred.append(predicted.item())

class_names = sorted(os.listdir("dataset/asl_alphabet_train"))

# macierz pomyłek
cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

# Rysujemy ją z etykietami i wartościami
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for ASL classifier")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from collections import Counter

errors = [ (true, pred) for true, pred in zip(y_true, y_pred) if true != pred ]
print("liczba sprawdzanych zdjęć:", len(y_true))
print("liczba dobrych wyborów:", len(y_true)-len(errors))
print("Liczba błędów:", len(errors))
print("Najczęstsze pomyłki:")
print(Counter(errors).most_common(10))

