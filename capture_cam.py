import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from landmark_extraction import get_face_landmarks


emotion_labels = {
    0: "Anger",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprise"
}




BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 50
NUM_CLASSES = 5
INPUT_SIZE = 1404
TEST_SPLIT = 0.1
DROPOUT_RATE = 0.5
def detect_face(image, gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    median_intensity = np.median(enhanced)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(enhanced, lower, upper)
    return cv2.resize(edges, (250, 250), interpolation=cv2.INTER_AREA)

def apply_canny_to_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detect_face(image, gray)

    if len(faces) == 0:
        print("No face detected.")
        return None

    x, y, w, h = faces[0]

    face_region = image[y:y + h, x:x + w]

    face_region_resized = cv2.resize(face_region, (250, 250), interpolation=cv2.INTER_AREA)

    edges = apply_canny(face_region_resized)

    return edges

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

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)




def real_time_emotion_detection(model, device, max_fps=10):
    model.eval()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    frame_delay = 1.0 / max_fps

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_face(frame, gray)

        if len(faces) == 0:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            x, y, w, h = faces[0]
            face_region = frame[y:y+h, x:x+w]

            if canny:
                edges = apply_canny(face_region)
                edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

                with torch.no_grad():
                    output = model(edges_tensor)
                    predicted_label = torch.argmax(output).item()
                    emotion = emotion_labels[predicted_label]
            else:
                face_landmarks = get_face_landmarks(face_region, draw=False)
                if len(face_landmarks) == 0:
                    continue
                if face_landmarks is None:
                    cv2.putText(frame, "No Landmarks Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    output = model(torch.tensor([face_landmarks], dtype=torch.float32).to(device))
                    print(output)
                    predicted_label = torch.argmax(output).item()
                    emotion = emotion_labels[predicted_label]

            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Real-Time Emotion Detection", frame)

        elapsed_time = time.time() - start_time
        remaining_time = frame_delay - elapsed_time

        if remaining_time > 0:
            # Sleep for the remaining time to maintain the frame rate
            time.sleep(remaining_time)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

canny = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if canny:
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("emotion_canny_model.pth"))
else:
    model = EmotionClassifier(INPUT_SIZE, 5).to(device)
    model.load_state_dict(torch.load("emotion_model.pth"))
# Start real-time emotion detection
real_time_emotion_detection(model, device)
