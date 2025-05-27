#resnet50_1_with_aug_bezZbTest
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Sprawdzenie czy jest GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# Ścieżki do danych
train_dir = r"C:\Users\zydor\Documents\projekt ASL ATLAS\data\asl_alphabet_train"
test_dir = r"C:\Users\zydor\Documents\projekt ASL ATLAS\data\asl_alphabet_test"

# Transformacje
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Wczytanie zbioru bez transformacji (placeholder, np. Resize)
base_transform = transforms.Compose([
    transforms.Resize((224, 224))  # neutralne, wspólne dla obu zbiorów
])
base_dataset = datasets.ImageFolder(train_dir, transform=base_transform)

# Podział na train i val
val_size = int(0.2 * len(base_dataset))
train_size = len(base_dataset) - val_size
train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size])

# Nadpisanie transformacji
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform

# Test set
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

# Loadery
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Załadowanie modelu ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)


# Zamrożenie feature extractor
for param in model.parameters():
    param.requires_grad = False

# Podmiana klasyfikatora
num_classes = len(train_dataset.dataset.classes)  # <-- ważne, bierzemy pełne klasy
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Funkcja kosztu i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Trenowanie modelu
num_epochs = 5
print("\nStart trenowania...\n")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    # Ewaluacja na zbiorze walidacyjnym
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / val_total

    print(f"Epoka {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

print(f"\nTrenowanie zakończone w {time.time() - start_time:.2f} sekund.")

# Ewaluacja na zbiorze testowym
print("\nEwaluacja na zbiorze testowym...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

# Zapisanie modelu
torch.save(model.state_dict(), 'resnet50_asl_aug1_3_data.pth')
