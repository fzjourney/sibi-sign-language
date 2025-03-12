import cv2
import torch
import torchvision.transforms as transforms
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np

MODEL_PATH = "Model/cnn_model.pth"
LABELS_PATH = "Model/labels.txt"
IMG_SIZE = 64

class SignLanguageCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 512) 
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x))) 
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN(len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(img):
    """ Convert image to RGB, resize, and normalize exactly like uploaded image. """
    img = Image.fromarray(img).convert("RGB")  
    img = transform(img).unsqueeze(0).to(device)
    return img

def predict_from_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)
        index = torch.argmax(prediction).item()
        class_name = class_names[index]
        confidence_score = torch.softmax(prediction, dim=1)[0][index] * 100

    print(f"Prediction: {class_name} ({confidence_score:.2f}%)")

camera = cv2.VideoCapture(1)
detector = HandDetector(detectionCon=0.8, maxHands=1)

prev_bbox = None
stabilization_threshold = 20  

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands, _ = detector.findHands(frame, draw=False)  

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        if prev_bbox is not None:
            dx = abs(x - prev_bbox[0])
            dy = abs(y - prev_bbox[1])
            dw = abs(w - prev_bbox[2])
            dh = abs(h - prev_bbox[3])

            if dx < stabilization_threshold and dy < stabilization_threshold and dw < stabilization_threshold and dh < stabilization_threshold:
                x, y, w, h = prev_bbox

        prev_bbox = (x, y, w, h)

        padding = 30
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)

        hand_img = frame_rgb[y1:y2, x1:x2]  
        if hand_img.size != 0:
            hand_tensor = preprocess_image(hand_img)  

            with torch.no_grad():
                prediction = model(hand_tensor)
                index = torch.argmax(prediction).item()
                class_name = class_names[index]
                confidence_score = torch.softmax(prediction, dim=1)[0][index] * 100

            text = f"{class_name} ({confidence_score:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.putText(frame, "Press 'I' to upload an image", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        predict_from_image()

camera.release()
cv2.destroyAllWindows()