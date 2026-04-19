import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.float().to(device)

        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            out = model(x).squeeze()
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).int()
            correct += (preds == y.int()).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total
