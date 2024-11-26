import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 50
NUM_CLASSES = 5
INPUT_SIZE = 1404
TEST_SPLIT = 0.1
DROPOUT_RATE = 0.5
class_names = ["Sad", "Surprise", "Anger", "Happy", "Neutral"]


# Создание Dataset
class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        scaler = StandardScaler()
        self.features = scaler.fit_transform(data.iloc[:, :-1].values.astype(np.float32))

        expression_map = {
            "Sad" : 0,
            "Surprise": 1,
            "Anger" :2,
            "Happy" : 3,
            "Neutral" : 4,
        }

        self.labels = data.iloc[:, -1].map(expression_map).fillna(-1).astype(np.int64)

        if np.any(self.labels == -1):
            print("Warning: Invalid labels found in the dataset.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)


class EmotionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Функция для обучения модели
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")


# Функция для тестирования модели
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

def visualize_errors(dataset, true_labels, predicted_labels):
    errors = [(i, t, p) for i, (t, p) in enumerate(zip(true_labels, predicted_labels)) if t != p]

    for idx, true, pred in errors[:10]:  # Показать первые 10 ошибок
        features, label = dataset[idx]
        print(f"True Label: {class_names[true]}, Predicted: {class_names[pred]}")
        print(f"Features: {features}")



def train_the_model():
    dataset = pd.read_csv("data.csv")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=y)

    rf_classifier = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy',
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    y_pred = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    with open('./best_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # dataset = EmotionDataset("data.csv")
    #
    # test_size = int(len(dataset) * TEST_SPLIT)
    # train_size = len(dataset) - test_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #
    # model = EmotionClassifier(INPUT_SIZE, NUM_CLASSES).to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #
    # train_model(model, train_loader, criterion, optimizer, device)
    #
    # print("Evaluating model on test data...")
    # evaluate_model(model, test_loader, device)
    #
    # torch.save(model.state_dict(), "emotion_model.pth")
    # print("Model saved!")
