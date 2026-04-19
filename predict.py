import torch
import os
from infer import load_model, predict_image
from utils.transforms import get_transforms
from utils.io import load_yaml

config = load_yaml("config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("./runs/exp_001/checkpoints/best_model.pth", device)
transform = get_transforms(config['data']['img_size'], False)

img_path = "your_image.jpg"

prob = predict_image(img_path, model, transform, device)

label = "Dog" if prob > config['inference']['threshold'] else "Cat"

print(f"Prediction: {label}, prob={prob:.4f}")
