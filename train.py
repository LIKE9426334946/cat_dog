import torch
from torch.utils.data import DataLoader
from model import CatDogCNN
from utils.seed import set_seed
from utils.io import load_yaml, ensure_dir
from utils.split import create_splits
from utils.dataset import CatDogDataset
from utils.transforms import get_transforms
from utils.trainer import train_one_epoch, eval_model
import os

config = load_yaml("config.yaml")

set_seed(config['project']['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_dir = os.path.join(config['paths']['runs_dir'], config['paths']['exp_name'])
ensure_dir(run_dir)
ensure_dir(os.path.join(run_dir, "splits"))
ensure_dir(os.path.join(run_dir, "checkpoints"))

train_df, val_df, test_df = create_splits(config, os.path.join(run_dir, "splits"))

train_dataset = CatDogDataset(os.path.join(run_dir,"splits/train.csv"),
                             get_transforms(config['data']['img_size'], True))
val_dataset = CatDogDataset(os.path.join(run_dir,"splits/val.csv"),
                           get_transforms(config['data']['img_size'], False))

train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

model = CatDogCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
criterion = torch.nn.BCEWithLogitsLoss()

best_acc = 0

for epoch in range(config['train']['epochs']):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch
    )

    val_loss, val_acc = eval_model(
        model, val_loader, criterion, device, epoch
    )

    print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(run_dir,"checkpoints/best_model.pth"))

torch.save(model.state_dict(), os.path.join(run_dir,"checkpoints/last_model.pth"))
