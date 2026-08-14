# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

import random
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

from ccrcc_grading.config import LABELS_FILE, MODEL_NAME, PATCH_FOLDER, SPLIT_FILE


def get_split_file_path(base_path, random_state):
    """
    Give each seed its own split file.

    The split is written to disk rather than recomputed so that every data form
    (RGB, HEC, modulated) trained under the same seed sees byte-identical
    train/val/test membership. Without this the preprocessing comparison would
    confound method with split.
    """
    base_path = Path(base_path)
    return base_path.parent / f"{base_path.stem}_seed{random_state}{base_path.suffix}"


def load_labels(labels_file):
    labels_df = pd.read_csv(labels_file)
    print(f"Total samples: {len(labels_df)}")

    if "Label" in labels_df.columns:
        print(f"Class distribution:\n{labels_df['Label'].value_counts()}")
    else:
        print("Warning: no 'Label' column in the labels file")

    return labels_df


def split_data_stratified(
    labels_df, label_column="Label", test_size=0.2, val_size=0.1, random_state=42
):
    """
    Stratified 70/10/20 train/validation/test split.

    Stratification is not optional here: grade 3 is roughly a tenth of the
    dataset and is the grade the method is meant to recover, so an unstratified
    split can leave the test set with too few grade 3 patches for balanced
    accuracy to mean anything.
    """
    train_val_df, test_df = train_test_split(
        labels_df,
        test_size=test_size,
        stratify=labels_df[label_column],
        random_state=random_state,
    )

    # val_size is a fraction of the whole dataset, so it has to be rescaled
    # against what is left after the test split is removed.
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df[label_column],
        random_state=random_state,
    )

    print(f"Train {len(train_df)}, validation {len(val_df)}, test {len(test_df)}")

    return train_df, val_df, test_df


def create_split_csv(labels_df, train_df, val_df, test_df, output_file):
    split_df = labels_df.copy()
    split_df["split"] = ""

    split_df.loc[train_df.index, "split"] = "train"
    split_df.loc[val_df.index, "split"] = "validation"
    split_df.loc[test_df.index, "split"] = "test"

    split_df.to_csv(output_file, index=False)
    print(f"Split assignments saved to {output_file}")

    return split_df


def get_label_mappings(labels_df, label_column="Label"):
    unique_labels = sorted(labels_df[label_column].unique())
    # Cast to native int, numpy integer types are not JSON serialisable and
    # these mappings end up in the MLflow artifact.
    label2id = {int(label): int(idx) for idx, label in enumerate(unique_labels)}
    id2label = {int(idx): int(label) for idx, label in enumerate(unique_labels)}

    print(f"Number of classes: {len(unique_labels)}")
    print(f"Label mappings: {label2id}")

    return label2id, id2label


def prepare_dataframe_with_paths(
    labels_df, images_dir, label2id, filename_column="Name", label_column="Label"
):
    df = labels_df.copy()
    df["image_path"] = df[filename_column].apply(lambda x: str(images_dir / x))

    df["exists"] = df["image_path"].apply(lambda x: Path(x).exists())
    missing_count = (~df["exists"]).sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} images not found and will be excluded")
    df = df[df["exists"]].drop("exists", axis=1)

    df["label_id"] = df[label_column].map(label2id)

    return df


def create_hf_dataset(
    df, processor, apply_flip_augmentation=False, random_state=42, force_flip=False
):
    """
    Build a HuggingFace dataset, optionally flipping images on load.

    force_flip is used to materialise an augmented copy of the minority classes.
    apply_flip_augmentation is the in-place variant that flips minority-class
    images without duplicating them; it is kept for ablations but is not the
    path used in the reported runs.
    """
    random.seed(random_state)

    majority_label_id = df["label_id"].value_counts().idxmax()

    def process_example(example):
        image = Image.open(example["image_path"]).convert("RGB")

        if force_flip:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif apply_flip_augmentation and example["label_id"] != majority_label_id:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

        inputs = processor(images=image, return_tensors="pt")
        example["pixel_values"] = inputs["pixel_values"].squeeze()
        example["labels"] = example["label_id"]
        return example

    dataset = Dataset.from_pandas(df[["image_path", "label_id"]])
    return dataset.map(process_example, remove_columns=["image_path"])


def create_datasets(
    train_df,
    val_df,
    test_df,
    processor,
    images_dir,
    label2id,
    filename_column="Name",
    label_column="Label",
    apply_flip_augmentation=False,
    random_state=42,
):
    """
    Build the three splits, augmenting only the training minority classes.

    The augmentation duplicates minority-class patches with a random flip
    rather than reweighting the loss, and touches the training split only.
    Validation and test keep the original imbalanced distribution, so balanced
    accuracy on them still reflects real class prevalence.
    """
    train_df = prepare_dataframe_with_paths(
        train_df, images_dir, label2id, filename_column, label_column
    )
    val_df = prepare_dataframe_with_paths(
        val_df, images_dir, label2id, filename_column, label_column
    )
    test_df = prepare_dataframe_with_paths(
        test_df, images_dir, label2id, filename_column, label_column
    )

    majority_label_id = train_df["label_id"].value_counts().idxmax()

    if apply_flip_augmentation:
        print("Flip augmentation on, minority classes duplicated")
        minority_df = train_df[train_df["label_id"] != majority_label_id]

        train_dataset_orig = create_hf_dataset(
            train_df,
            processor,
            apply_flip_augmentation=False,
            random_state=random_state,
        )
        train_dataset_flip = create_hf_dataset(
            minority_df,
            processor,
            apply_flip_augmentation=False,
            random_state=random_state,
            force_flip=True,
        )
        train_dataset = concatenate_datasets([train_dataset_orig, train_dataset_flip])
    else:
        print("Flip augmentation off")
        train_dataset = create_hf_dataset(
            train_df,
            processor,
            apply_flip_augmentation=False,
            random_state=random_state,
        )

    val_dataset = create_hf_dataset(
        val_df, processor, apply_flip_augmentation=False, random_state=random_state
    )
    test_dataset = create_hf_dataset(
        test_df, processor, apply_flip_augmentation=False, random_state=random_state
    )

    print(
        f"Dataset sizes: train {len(train_dataset)}, "
        f"validation {len(val_dataset)}, test {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df


def get_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size=16, num_workers=4
):
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    # pin_memory is a CUDA feature and is a no-op or worse on MPS.
    use_pin_memory = torch.cuda.is_available()

    loaders = []
    for dataset, shuffle in (
        (train_dataset, True),
        (val_dataset, False),
        (test_dataset, False),
    ):
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=use_pin_memory,
            )
        )

    train_loader, val_loader, test_loader = loaders
    print(
        f"DataLoader batches: train {len(train_loader)}, "
        f"validation {len(val_loader)}, test {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader


def prepare_data_pipeline(
    images_dir,
    labels_file,
    model_name,
    test_size=0.2,
    val_size=0.1,
    batch_size=16,
    num_workers=4,
    random_state=42,
    filename_column="Name",
    label_column="Label",
    apply_flip_augmentation=False,
):
    """
    Load labels, resolve or create the seed split, and build datasets.

    An existing split file for the given seed is always reused rather than
    regenerated. This is what makes runs across data forms comparable, and it
    also means deleting a split file silently invalidates comparability with
    every earlier run that used it.
    """
    labels_df = load_labels(labels_file)
    label2id, id2label = get_label_mappings(labels_df, label_column=label_column)

    split_file = get_split_file_path(SPLIT_FILE, random_state)

    if split_file.exists():
        print(f"Using existing split {split_file} (seed {random_state})")
        split_df = pd.read_csv(split_file)

        train_df = split_df[split_df["split"] == "train"].copy()
        val_df = split_df[split_df["split"] == "validation"].copy()
        test_df = split_df[split_df["split"] == "test"].copy()
    else:
        print(f"No split found for seed {random_state}, creating one")
        train_df, val_df, test_df = split_data_stratified(
            labels_df,
            label_column=label_column,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
        create_split_csv(labels_df, train_df, val_df, test_df, split_file)

    processor = ViTImageProcessor.from_pretrained(model_name)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_df_final,
        val_df_final,
        test_df_final,
    ) = create_datasets(
        train_df,
        val_df,
        test_df,
        processor,
        images_dir,
        label2id,
        filename_column=filename_column,
        label_column=label_column,
        apply_flip_augmentation=apply_flip_augmentation,
        random_state=random_state,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return {
        "datasets": {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        },
        "dataloaders": {
            "train": train_loader,
            "validation": val_loader,
            "test": test_loader,
        },
        "dataframes": {
            "train": train_df_final,
            "validation": val_df_final,
            "test": test_df_final,
        },
        "label_mappings": {
            "label2id": label2id,
            "id2label": id2label,
        },
        "processor": processor,
        "num_classes": len(label2id),
        "random_seed": random_state,
        "test_size": test_size,
        "val_size": val_size,
        "apply_flip_augmentation": apply_flip_augmentation,
    }


if __name__ == "__main__":
    data = prepare_data_pipeline(
        images_dir=PATCH_FOLDER,
        labels_file=LABELS_FILE,
        model_name=MODEL_NAME,
        batch_size=16,
        num_workers=4,
        random_state=313,
        apply_flip_augmentation=True,
    )

    print(f"Number of classes: {data['num_classes']}")
    print(f"Label mappings: {data['label_mappings']['label2id']}")
    print(f"Seed: {data['random_seed']}")
