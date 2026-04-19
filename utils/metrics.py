import torch

def accuracy(preds, labels):
    preds = (torch.sigmoid(preds) > 0.5).int()
    return (preds.squeeze() == labels).float().mean().item()
