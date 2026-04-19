import yaml
import os
import pandas as pd

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f)

def save_csv(df, path):
    df.to_csv(path, index=False)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
