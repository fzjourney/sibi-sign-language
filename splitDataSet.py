import os
import shutil
import random
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

SOURCE_DIR = "Data"
TRAIN_DIR = "Dataset/train"
VAL_DIR = "Dataset/validation"
TEST_DIR = "Dataset/test"
SPLIT_RATIO = {"train": 0.7, "validation": 0.15, "test": 0.15}

def create_dirs():
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(split, exist_ok=True)
        for label in os.listdir(SOURCE_DIR):
            os.makedirs(os.path.join(split, label), exist_ok=True)

def split_data():
    create_dirs()
    for label in os.listdir(SOURCE_DIR):
        images = os.listdir(os.path.join(SOURCE_DIR, label))
        random.shuffle(images)
        train_end = int(len(images) * SPLIT_RATIO["train"])
        val_end = train_end + int(len(images) * SPLIT_RATIO["validation"])

        for i, img in enumerate(images):
            src_path = os.path.join(SOURCE_DIR, label, img)
            if i < train_end:
                dest_path = os.path.join(TRAIN_DIR, label, img)
            elif i < val_end:
                dest_path = os.path.join(VAL_DIR, label, img)
            else:
                dest_path = os.path.join(TEST_DIR, label, img)
            shutil.copy(src_path, dest_path)
    print("Dataset successfully split!")

class SignLanguageCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

with open("Model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN(len(class_names)).to(device)
model.load_state_dict(torch.load("Model/cnn_model.pth", map_location=device))
model.eval()

def classify_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
        index = torch.argmax(prediction).item()
        class_name = class_names[index]
        confidence_score = torch.softmax(prediction, dim=1)[0][index] * 100
    
    print(f"Predicted: {class_name} ({confidence_score:.2f}%)")
    return class_name, confidence_score

if __name__ == "__main__":
    split_data()
    test_image = input("Enter image path to test: ")
    if os.path.exists(test_image):
        classify_image(test_image)
    else:
        print("Invalid image path.")
