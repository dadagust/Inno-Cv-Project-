import csv
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from landmark_extraction import get_face_landmarks



def process_image(image_path, device="cuda"):
    """
    Считывает изображение, изменяет размер и возвращает landmarks.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    with torch.no_grad():
        image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=torch.float32, device=device)

    return image

def process_labels(data_dirs, label_files, output_file):
    rows = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for data_dir, label_file in zip(data_dirs, label_files):
        label_df = pd.read_csv(label_file)

        for _, row in tqdm(label_df.iterrows(), desc=f"Processing images from {data_dir}", total=label_df.shape[0]):
            image_name = row['filename']
            expression_label = str(row['class'])

            image_path = os.path.join(data_dir, image_name)

            if not os.path.exists(image_path):
                continue

            try:
                standardized_face = process_image(image_path, device=device)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

            image_np = standardized_face.cpu().numpy().astype(np.uint8)
            del standardized_face
            torch.cuda.empty_cache()

            landmarks = get_face_landmarks(image_np)
            if not landmarks:
                continue

            row_data = landmarks + [expression_label]
            rows.append(row_data)

    # Записываем данные в CSV
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        header = [f"point{i + 1}" for i in range(len(rows[0]) - 1)] + ["expression_label"]
        csv_writer.writerow(header)
        csv_writer.writerows(rows)

    print(f"Data saved to {output_file}")

def prepare_data():
    train_data_dir = "Emotion detection.v2i.tensorflow/train"
    train_label_file = "Emotion detection.v2i.tensorflow/train/_annotations.csv"

    valid_data_dir = "Emotion detection.v2i.tensorflow/valid"
    valid_label_file = "Emotion detection.v2i.tensorflow/valid/_annotations.csv"

    output_file = "data.csv"

    process_labels([train_data_dir, valid_data_dir], [train_label_file, valid_label_file], output_file)
    df = pd.read_csv(output_file)

    unique_labels = df['expression_label'].unique()
    num_rows = df.shape[0]

    print(f"Number of rows: {num_rows}")

    print("Unique classes in 'expression_label':")
    for label in unique_labels:
        print(label)

    num_classes = len(unique_labels)
    print(f"Total number of unique classes: {num_classes}")

    label_counts = df['expression_label'].value_counts()
    print("Number of elements per class:")
    for label, count in label_counts.items():
        print(f"Class '{label}': {count}")