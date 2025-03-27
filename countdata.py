import os

train_dir = "Dataset/train"
val_dir = "Dataset/validation"

train_count = sum(len(os.listdir(os.path.join(train_dir, label))) for label in os.listdir(train_dir))
val_count = sum(len(os.listdir(os.path.join(val_dir, label))) for label in os.listdir(val_dir))

print(f"Train folder: {train_count} images")
print(f"Validation folder: {val_count} images")