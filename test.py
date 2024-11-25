import cv2
import numpy as np

from landmark_extraction import get_face_landmarks

from PIL import Image

image_path = './Emotion recognition.v2i.tensorflow/train/image0000011_jpg.rf.978289ed7829dfa54e754269ad26faaa.jpg'

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
with Image.open(image_path) as image:
    cropped_array = np.array(img)

try:
    face_landmarks = get_face_landmarks(img, draw=True)
    print(face_landmarks)
except Exception as e:
    raise ValueError(f"Error extracting landmarks: {e}")

# Сохраняем обработанное изображение
output_path = './output_image_with_landmarks.jpg'
cv2.imwrite(output_path, img)
print(f"Image saved: {output_path}")