# Facial Emotion Recognition 

The primary goal of this project is to develop an accurate system capable of recognizing emotions from facial images. The
ability to identify emotional states has significant implications for various industries: in healthcare, it can help diagnose mental
health conditions; in customer service, it can improve user experience by adjusting responses based on customer emotions; in
entertainment, it can enhance gaming experiences by creating adaptive storylines; and in security, it can provide insights into
potential threats based on individuals’ emotional responses.

It is implemented in two different ways: Canny edge and Face mesh approaches

The pipeline is the follows:

> For canny edge: The canny procedure is applied on the image via apply_canny() function, then the extracted image features are forwarded to the model.

> For face mesh: We are using face_mesh = mp.solutions.face_mesh.FaceMesh() in the landmark_extraction.py. get_face_landmark function extracts 468 points, with x, y and z coordinates for each, then this data is forwarded to the RandomForestClassifier()

If you want to visualize the landmarks extracted by get_face_landmarks function, you can set draw=True, which will draw
the landmarks on the provided image. For photos, you need to use static_image_mode=False, for dynamic content you need True


Also, we have two python test files, for each approach. 

> To test a Canny approach you can run test_canny_model.py script, where you need to specify path to the image via image_path variable
 
> If you want to test Face mesh, you need to run test.py script and provide path to the image as in the Canny method.

You can just run main.py script, that will preprocess data and train the model

weights for both models can be downloaded here: https://disk.yandex.ru/d/qeS6rFYpQOQqGg
