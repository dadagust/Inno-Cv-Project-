import pickle
import cv2
from landmark_extraction import get_face_landmarks

with open('./model', 'rb') as f:
    model = pickle.load(f)

image_path = './MV5BYTAzY2Q1OTYtMzMwZS00NDNhLTliNjYtYjAxYzEyYzFmY2JjXkEyXkFqcGd.jpg'
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


face_landmarks = get_face_landmarks(img, draw=True)
if face_landmarks:
    output_path = './output_image_with_landmarks.jpg'
    cv2.imwrite(output_path, img)
    print(f"Image {output_path} saved")

    output = model.predict([face_landmarks])
    print(output)
    print(output[0])
else:
    print("There is no landmarks")
