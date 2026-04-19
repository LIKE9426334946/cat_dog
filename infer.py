import torch
from PIL import Image
from model import CatDogCNN
from utils.transforms import get_transforms

def load_model(path, device):
    model = CatDogCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(img_path, model, transform, device):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.sigmoid(out).item()

    return prob
