# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

import json
import shutil

import mlflow
import pandas as pd
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from ccrcc_grading.config import (
    CLASSMAP_FOLDER,
    DATA_PATH,
    HEC_FOLDER,
    LABELS_FILE,
    ML_RUNS,
    MODEL_NAME,
    OUTPUT_DIR,
    PATCH_FOLDER,
    SOURCE_CLASSMAP_FOLDER,
    SOURCE_HEC_FOLDER,
    SOURCE_LABELS_FOLDER,
    SOURCE_PATCH_FOLDER,
    SPLIT_FILE,
)
from ccrcc_grading.data_handler import get_split_file_path, prepare_data_pipeline
from ccrcc_grading.evaluation import (
    compute_metrics,
    evaluate_and_log,
    plot_training_history,
)
from ccrcc_grading.modulation import generate_config_name, make_modulated_dataset
from ccrcc_grading.perturbation import introduce_classification_errors

# Label used in run names and cached dataset folders for the combined
# segmentation plus classification perturbation.
PERTURBATION_TYPE = "XX"

STATIC_DATA_FORMS = {
    "patches": (PATCH_FOLDER, SOURCE_PATCH_FOLDER),
    "classmap": (CLASSMAP_FOLDER, SOURCE_CLASSMAP_FOLDER),
    "hec": (HEC_FOLDER, SOURCE_HEC_FOLDER),
}


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def resolve_images_path(
    data_form, modulation_configs, perturbation_configs, split_file_path, random_seed
):
    """
    Return the directory of images to train on, generating it if absent.

    Static forms are copied from the read-only source once per run and deleted
    afterwards, since the working copy can be tens of gigabytes. Modulated
    forms are generated on demand and cached under a name derived from the
    full configuration, so an identical configuration is never rebuilt and a
    changed one never silently reuses stale pixels.
    """
    if data_form in STATIC_DATA_FORMS:
        images_path, source_path = STATIC_DATA_FORMS[data_form]
        if not images_path.exists():
            print(f"Copying {data_form} from {source_path} to {images_path}")
            shutil.copytree(source_path, images_path)
        return images_path, None

    if data_form != "mix":
        raise ValueError(
            f"Invalid data_form: {data_form}. "
            "Expected one of 'patches', 'classmap', 'hec', 'mix'"
        )

    config_name = generate_config_name(modulation_configs)
    labels_path = SOURCE_LABELS_FOLDER

    if perturbation_configs is not None:
        config_extra = (
            f"Per_{perturbation_configs['type']}_{int(perturbation_configs['level'])}"
        )
        config_name = f"{config_extra}_{config_name}"
        labels_path = DATA_PATH / config_extra

        if not labels_path.exists() or not any(labels_path.iterdir()):
            print(f"Building perturbed class maps: {perturbation_configs}")
            introduce_classification_errors(
                error_percentage=perturbation_configs["level"],
                csv_path=split_file_path,
                mat_folder_path=SOURCE_LABELS_FOLDER,
                output_path=labels_path,
                use_smart_transitions=True,
                random_seed=random_seed,
            )

    images_path = DATA_PATH / config_name
    if not images_path.exists():
        make_modulated_dataset(
            modulation_configs,
            SOURCE_PATCH_FOLDER,
            labels_path,
            images_path,
        )

    return images_path, config_name


def train(
    data_form="patches",
    train_epochs=1,
    random_seed=42,
    apply_flip_augmentation=False,
    modulation_configs=None,
    clean_up=True,
    perturbation_configs=None,
):
    """
    Fine-tune the ViT on one data form and log the run to MLflow.

    Every preprocessing variant goes through this same function with the same
    hyperparameters and the same seed split. That is deliberate: the comparison
    in the paper is between input representations, so anything else that could
    move the metrics has to be held fixed.
    """
    ML_RUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ML_RUNS}/mlflow.db")
    mlflow.set_experiment(f"cell-grading-vit-{data_form}")

    with mlflow.start_run(run_name=f"vit_training_{data_form}_seed{random_seed}"):
        device, device_name = select_device()
        print(f"Using device: {device_name}")
        mlflow.log_param("device", device_name)

        split_file_path = get_split_file_path(SPLIT_FILE, random_seed)

        images_path, config_name = resolve_images_path(
            data_form,
            modulation_configs,
            perturbation_configs,
            split_file_path,
            random_seed,
        )

        if config_name is not None:
            mlflow.log_param("mixer_configs", modulation_configs)
            mlflow.log_param("config_name", config_name)

        data = prepare_data_pipeline(
            images_dir=images_path,
            labels_file=LABELS_FILE,
            model_name=MODEL_NAME,
            random_state=random_seed,
            apply_flip_augmentation=apply_flip_augmentation,
        )

        mlflow.log_artifact(str(split_file_path), "datasets")

        splits_df = pd.read_csv(split_file_path)
        mlflow.log_params(
            {
                "total_samples": len(splits_df),
                "train_samples": len(splits_df[splits_df["split"] == "train"]),
                "val_samples": len(splits_df[splits_df["split"] == "validation"]),
                "test_samples": len(splits_df[splits_df["split"] == "test"]),
                "data_form": data_form,
                "model_name": MODEL_NAME,
                "split_file": str(split_file_path),
                "random_seed": data["random_seed"],
                "data_split_test_size": data["test_size"],
                "data_split_val_size": data["val_size"],
                "apply_flip_augmentation": data["apply_flip_augmentation"],
                "perturbation_type": (
                    perturbation_configs.get("type", "NA")
                    if perturbation_configs
                    else "NA"
                ),
                "perturbation_level": (
                    perturbation_configs.get("level", 0) if perturbation_configs else 0
                ),
            }
        )

        train_dataset = data["datasets"]["train"]
        val_dataset = data["datasets"]["validation"]
        test_dataset = data["datasets"]["test"]
        label2id = data["label_mappings"]["label2id"]
        id2label = data["label_mappings"]["id2label"]
        num_classes = data["num_classes"]

        mlflow.log_params(
            {
                "num_classes": num_classes,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
            }
        )

        image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

        # ignore_mismatched_sizes lets the pretrained 21k-class head be dropped
        # and replaced with a three-grade head.
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model.to(device)

        output_dir_data = OUTPUT_DIR / data_form
        output_dir_data.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir_data),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=str(output_dir_data / "logs"),
            logging_steps=10,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=["mlflow"],
            warmup_ratio=0.1,
            run_name=f"vit_{data_form}",
            logging_first_step=True,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        mlflow.log_params(
            {
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                "weight_decay": training_args.weight_decay,
                "warmup_ratio": training_args.warmup_ratio,
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        plot_training_history(trainer, data_form)

        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": trainer.model,
                "image_processor": image_processor,
            },
            name=f"vit-{data_form}-classifier",
            task="image-classification",
            metadata={
                "model_name": MODEL_NAME,
                "num_classes": num_classes,
                "data_form": data_form,
            },
        )

        val_metrics = evaluate_and_log(
            trainer, val_dataset, id2label, "validation", data_form
        )
        test_metrics = evaluate_and_log(
            trainer, test_dataset, id2label, "test", data_form
        )

        all_metrics = {
            "data_form": data_form,
            "validation": val_metrics,
            "test": test_metrics,
            "num_classes": num_classes,
            "label2id": label2id,
            "id2label": {int(k): v for k, v in id2label.items()},
        }

        metrics_path = output_dir_data / f"{data_form}_evaluation.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        mlflow.log_artifact(str(metrics_path), "metrics")

        if images_path.exists() and clean_up:
            print(f"Removing working image directory {images_path}")
            shutil.rmtree(images_path)

        print(f"Model logged to {model_info.model_uri}")
        print(f"MLflow UI: mlflow ui --backend-store-uri sqlite:///{ML_RUNS}/mlflow.db")


def build_experiment_list(seeds, modulation_configs, perturbation_levels):
    """
    Enumerate the runs behind the reported tables.

    Two blocks. The first sweeps every preprocessing variant once on a single
    seed, which produces the method comparison. The second repeats only the top
    three modulation configurations across all seeds and perturbation levels,
    which produces the sensitivity curves; running the full config grid at
    every level was not worth the compute.
    """
    experiments = [
        [seeds[0], "patches", None, None],
        [seeds[0], "classmap", None, None],
        [seeds[0], "hec", None, None],
    ]

    for config in modulation_configs:
        experiments.append([seeds[0], "mix", config, None])

    for seed in seeds:
        for config in modulation_configs[:3]:
            for level in perturbation_levels:
                experiments.append(
                    [
                        seed,
                        "mix",
                        config,
                        {"type": PERTURBATION_TYPE, "level": level},
                    ]
                )

    return experiments


if __name__ == "__main__":
    from ccrcc_grading.modulation_configs import MODULATION_CONFIGS

    seeds = [7, 12, 14, 313]
    perturbation_levels = list(range(10, 110, 10))

    for seed, data_form, modulation_config, perturbation_config in (
        build_experiment_list(seeds, MODULATION_CONFIGS, perturbation_levels)
    ):
        train(
            data_form=data_form,
            train_epochs=10,
            random_seed=seed,
            apply_flip_augmentation=True,
            modulation_configs=modulation_config,
            clean_up=True,
            perturbation_configs=perturbation_config,
        )
