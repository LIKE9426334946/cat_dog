import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for x, y in pbar:
        x, y = x.to(device), y.float().to(device)

        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    return total_loss / len(loader)

def eval_model(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            out = model(x).squeeze()
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).int()
            correct += (preds == y.int()).sum().item()
            total += y.size(0)
            acc = correct / total
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}"
            })

    return total_loss / len(loader), correct / total
