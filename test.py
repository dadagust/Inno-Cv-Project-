import cv2
import numpy as np
import torch

from landmark_extraction import get_face_landmarks


def process_image(image_path, crop_coords, device="cuda"):
    """
    Считывает изображение, обрезает его по заданным координатам и возвращает numpy массив.

    Args:
        image_path (str): Путь к изображению.
        crop_coords (tuple): Координаты обрезки в формате (left, top, right, bottom).
        device (str): Устройство для размещения тензора ("cuda" или "cpu").

    Returns:
        np.ndarray: Обрезанное изображение в виде numpy массива.
    """
    # Считываем изображение
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Загружаем в формате BGR
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Преобразуем BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Извлекаем координаты для обрезки
    left, top, right, bottom = crop_coords

    # Обрезаем изображение
    cropped_image = image[top:bottom, left:right]

    return cropped_image


# Путь к изображению
image_path = './data2/archive/origin/angry_american_25.jpg'
crop_coords = (538, 76, 1461, 999)  # Координаты обрезки (left, top, right, bottom)

# Обработка изображения
image_np = process_image(image_path, crop_coords, device="cuda")

print("Shape of processed image:", image_np.shape)

# Преобразуем изображение обратно в uint8 (если оно было нормализовано ранее)
image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)

try:
    # Получаем landmarks с обработанного изображения
    face_landmarks = get_face_landmarks(image_uint8, draw=True)
    print(face_landmarks)
except Exception as e:
    raise ValueError(f"Ошибка при извлечении landmarks: {e}")

# Сохраняем обработанное изображение
output_path = './output_image_with_landmarks.jpg'
cv2.imwrite(output_path, cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))  # Сохраняем в формате BGR
print(f"Обработанное изображение сохранено: {output_path}")
