import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        label = row['label']

        if self.transform:
            img = self.transform(img)

        return img, label
