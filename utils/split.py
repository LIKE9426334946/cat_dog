import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def collect_jpg_images(folder, label, class_name):
    records = []
    invalid_files = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")

    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)

        if not os.path.isfile(fpath):
            continue

        if not fname.lower().endswith(".jpg"):
            continue

        if not is_valid_image(fpath):
            invalid_files.append(fpath)
            continue

        records.append([fpath, label, class_name])

    return records, invalid_files


def create_splits(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    data_root = config["paths"]["data_root"]
    cat_dir = os.path.join(data_root, config["paths"]["cat_dir"])
    dog_dir = os.path.join(data_root, config["paths"]["dog_dir"])

    seed = config["project"]["seed"]

    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    test_ratio = config["data"]["test_ratio"]

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
        )

    cat_label = config["data"]["label_map"]["Cat"]
    dog_label = config["data"]["label_map"]["Dog"]

    cat_records, cat_invalid = collect_jpg_images(cat_dir, cat_label, "Cat")
    dog_records, dog_invalid = collect_jpg_images(dog_dir, dog_label, "Dog")

    records = cat_records + dog_records
    invalid_files = cat_invalid + dog_invalid

    if len(records) == 0:
        raise RuntimeError("No valid .jpg images found in dataset.")

    df = pd.DataFrame(records, columns=["filepath", "label", "class_name"])

    print("=" * 60)
    print("Dataset scan finished")
    print(f"Valid Cat images : {len(cat_records)}")
    print(f"Valid Dog images : {len(dog_records)}")
    print(f"Invalid images   : {len(invalid_files)}")
    print(f"Total valid imgs : {len(df)}")
    print("=" * 60)

    if len(invalid_files) > 0:
        invalid_df = pd.DataFrame({"invalid_filepath": invalid_files})
        invalid_df.to_csv(os.path.join(save_dir, "invalid_images.csv"), index=False)
        print(f"Invalid image list saved to: {os.path.join(save_dir, 'invalid_images.csv')}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=seed,
        shuffle=True
    )

    val_relative_ratio = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        stratify=temp_df["label"],
        random_state=seed,
        shuffle=True
    )

    train_path = os.path.join(save_dir, "train.csv")
    val_path = os.path.join(save_dir, "val.csv")
    test_path = os.path.join(save_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Split finished")
    print(f"Train: {len(train_df)} -> {train_path}")
    print(f"Val  : {len(val_df)} -> {val_path}")
    print(f"Test : {len(test_df)} -> {test_path}")
    print("=" * 60)

    return train_df, val_df, test_df
