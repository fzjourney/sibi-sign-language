import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import time
from cvzone.HandTrackingModule import HandDetector
from PIL import Image

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.5)

img_size = 300  
offset = 20  

letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'J']
folder_index = 0  
counter = 0  

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

while folder_index < len(letters):
    folder = f"Data/{letters[folder_index]}"
    os.makedirs(folder, exist_ok=True)

    success, img = cap.read()
    if not success:
        print("❌ Error: Failed to read frame from webcam.")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        img_crop = img[y1:y2, x1:x2]
        if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
            print("⚠️ Warning: Cropped image is empty, skipping.")
            continue

        img_pil = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil)

        img_numpy = np.transpose(img_tensor.numpy(), (1, 2, 0))
        img_numpy = (img_numpy * 255).astype(np.uint8)

        cv2.imshow("Processed Image", img_numpy)
    
    cv2.imshow("Webcam Feed", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        img_path = f"{folder}/Image_{time.time()}.jpg"
        img_pil.save(img_path)
        print(f"✅ Saved {img_path} ({counter} images in {letters[folder_index]})")

        if counter >= 5:
            counter = 0  
            folder_index += 1  
            if folder_index >= len(letters):
                print("✅ Dataset collection completed.")
                break
    elif key == ord("q"):
        print("🔴 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
