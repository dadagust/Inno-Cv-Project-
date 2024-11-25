import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

def apply_canny(image, region):
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_region = clahe.apply(gray_region)

    median_intensity = np.median(enhanced_region)
    lower_threshold = int(max(0, 0.66 * median_intensity))
    upper_threshold = int(min(255, 1.33 * median_intensity))

    edges = cv2.Canny(enhanced_region, lower_threshold, upper_threshold)

    return edges, region


class EmotionDatasetCOCO(Dataset):
    def __init__(self, dataset_dir, split="train", transform=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        annotations_file = os.path.join(dataset_dir, split, "_annotations.coco.json")
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        self.samples = []

        for annotation in self.annotations["annotations"]:
            image_id = annotation["image_id"]
            bbox = annotation["bbox"]
            label = annotation["category_id"]

            for img in self.annotations["images"]:
              if img["id"] == image_id:
                image_info = img

            image_path = os.path.join(dataset_dir, split, image_info["file_name"])
            self.samples.append((image_path, bbox, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      image_path, bbox, label = self.samples[idx]
      image = cv2.imread(image_path)
      if image is None:
          raise FileNotFoundError(f"Image {image_path} not found.")

      _, edges = self.preprocess_face(image, bbox)

      label -= 1

      # Apply transformations
      if self.transform:
          edges = self.transform(edges)

      return edges, label


    @staticmethod
    def preprocess_face(image, bbox):
        x, y, w, h = map(int, bbox)
        face_region = image[y:y+h, x:x+w]

        edges, face_region = apply_canny(image, face_region)

        edges = cv2.resize(edges, (250, 250), interpolation=cv2.INTER_AREA)

        return face_region, edges


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#dataset can be downloaded from https://universe.roboflow.com/detection-algorithm-comparison/emotion-detection-tgdyh/dataset/2
#here we use COCO format
dataset_dir = "/content/Emotion-detection-2"

train_dataset = EmotionDatasetCOCO(dataset_dir, split="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

valid_dataset = EmotionDatasetCOCO(dataset_dir, split="valid", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 62 * 62)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_canny(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), "emotion_canny_model.pth")
        print("model saved")
        print("Evaluating model on test data...")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")




