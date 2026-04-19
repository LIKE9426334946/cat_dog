import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(config, save_dir):
    data_root = config['paths']['data_root']
    cat_dir = os.path.join(data_root, config['paths']['cat_dir'])
    dog_dir = os.path.join(data_root, config['paths']['dog_dir'])

    records = []

    for img in os.listdir(cat_dir):
        records.append([os.path.join(cat_dir, img), 0, "Cat"])
    for img in os.listdir(dog_dir):
        records.append([os.path.join(dog_dir, img), 1, "Dog"])

    df = pd.DataFrame(records, columns=["filepath","label","class"])

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])

    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    return train_df, val_df, test_df
