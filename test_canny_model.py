import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_labels = {
    0: "Anger",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprise"
}

def detect_face(image, gray):
    """
    Detect faces in the image using Haar cascades.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def apply_canny_to_face(image):
    """
    Apply Canny edge detection to the detected face region.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detect_face(image, gray)

    if len(faces) == 0:
        print("No face detected.")
        return None, None

    x, y, w, h = faces[0]
    face_region = image[y:y + h, x:x + w]

    face_region_resized = cv2.resize(face_region, (250, 250), interpolation=cv2.INTER_AREA)

    gray_region = cv2.cvtColor(face_region_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_region = clahe.apply(gray_region)

    median_intensity = np.median(enhanced_region)
    lower_threshold = int(max(0, 0.66 * median_intensity))
    upper_threshold = int(min(255, 1.33 * median_intensity))

    edges = cv2.Canny(enhanced_region, lower_threshold, upper_threshold)

    return face_region_resized, edges

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

model = EmotionCNN().to(device)
print("Loading emotion_canny_model.pth")
model.load_state_dict(torch.load("emotion_canny_model.pth"))

def test_image(image_path, model, device):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"Original image shape: {original_image.shape}")

    face_region, edges = apply_canny_to_face(original_image)

    if edges is None:
        print("Skipping prediction due to no detected face.")
        return

    edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    model.eval()
    with torch.no_grad():
        output = model(edges_tensor)
        predicted_label = torch.argmax(output).item()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Face Region")
    axs[1].axis("off")

    axs[2].imshow(edges, cmap="gray")
    axs[2].set_title("Canny Edge")
    axs[2].axis("off")

    plt.suptitle(f"Prediction: {emotion_labels[predicted_label]}")
    plt.show()
    print("Model Output:", output)


#out image into folder and change name here
image_path = "sample.jpg"
test_image(image_path, model, device)
