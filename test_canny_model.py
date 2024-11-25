import random
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn
import torch.nn.functional as F


emotion_labels = {
    0: "Anger",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprise"
}

def apply_canny(image, region):
      gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      enhanced_region = clahe.apply(gray_region)

      median_intensity = np.median(enhanced_region)
      lower_threshold = int(max(0, 0.66 * median_intensity))
      upper_threshold = int(min(255, 1.33 * median_intensity))

      edges = cv2.Canny(enhanced_region, lower_threshold, upper_threshold)

      return edges, region

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

print("loading emotion_canny_model.pth")
model.load_state_dict(torch.load("emotion_canny_model.pth"))

def test_image(image_path, model, device):
    """
    Test the model on a single image from a provided path.
    """
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"Original image shape: {original_image.shape}")

    # Generate a dummy bounding box assuming the entire image is the face region
    bbox = [0, 0, original_image.shape[1], original_image.shape[0]]

    # Preprocess the image
    face_region, edges = EmotionDatasetCOCO.preprocess_face(original_image, bbox)

    # Convert edges to a tensor and normalize
    edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # Pass the preprocessed tensor to the model
    model.eval()
    with torch.no_grad():
        output = model(edges_tensor)
        predicted_label = torch.argmax(output).item()

    # Display the results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(edges, cmap="gray")
    axs[1].set_title("Canny Edge")
    axs[1].axis("off")

    plt.suptitle(f"Prediction: {emotion_labels[predicted_label]}")
    plt.show()
    print("Model Output:", output)





image_path = "sample.jpg"

test_image(image_path, model, device)