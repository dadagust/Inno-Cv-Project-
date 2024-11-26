# import cv2
# import numpy as np
# import torch
# from torch import nn
# from landmark_extraction import get_face_landmarks
# from PIL import Image
# from train_model import INPUT_SIZE, DROPOUT_RATE
#
# emotion_labels = {
#     0: "Sad",
#     1: "Surprise",
#     2: "Anger",
#     3: "Happy",
#     4: "Neutral",
# }
# emotions = ['Sad', 'Surprise', 'Anger', 'Happy', 'Neutral']
# class EmotionClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(EmotionClassifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(DROPOUT_RATE),
#
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(DROPOUT_RATE),
#
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(DROPOUT_RATE),
#
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, x):
#         return self.fc(x)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = EmotionClassifier(INPUT_SIZE, 5).to(device)
# model.load_state_dict(torch.load("emotion_model.pth"))
#
# image_path = './Emotion detection.v2i.tensorflow/train/image0013532_jpg.rf.02345a9420bde32d9e3f22a80fc85c89.jpg'
#
# img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# with Image.open(image_path) as image:
#     cropped_array = np.array(img)
#
# try:
#     face_landmarks = get_face_landmarks(img, draw=False)
# except Exception as e:
#     raise ValueError(f"Error extracting landmarks: {e}")
#
# # Сохраняем обработанное изображение
# output_path = './output_image_with_landmarks.jpg'
# cv2.imwrite(output_path, img)
# print(f"Image saved: {output_path}")
# model.eval()
# output = model(torch.tensor([face_landmarks], dtype=torch.float32).to(device))
# print(output)
# predicted_label = torch.argmax(output).item()
# print(predicted_label)
# emotion = emotion_labels[predicted_label]
# print(emotion)
#
import pickle

import cv2

from landmark_extraction import get_face_landmarks

with open('./model', 'rb') as f:
    model = pickle.load(f)

image_path = './Emotion detection.v2i.tensorflow/train/image0003664_jpg.rf.0564e38ccb61321664b39be39068ae3c.jpg'

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
emotions = ['Sad', 'Surprise', 'Anger', 'Happy', 'Neutral']

face_landmarks = get_face_landmarks(img, draw=False)

output = model.predict([face_landmarks])
print(output)
print(output[0])