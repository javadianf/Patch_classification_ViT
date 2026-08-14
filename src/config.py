# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

import os
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]

# Working directory. Each experiment copies its image set here and deletes it
# afterwards, so this should be fast local storage, not a network mount.
DATA_PATH = Path(os.environ.get("CCRCC_DATA", BASE_PATH / "data"))

# Read-only source of the raw dataset. Originally a mounted share; any directory
# with the same Patches / Labels / Classmap / HEC layout will work.
SOURCE_DATA = Path(os.environ.get("CCRCC_SOURCE_DATA", DATA_PATH / "raw"))

LABELS_FILE = DATA_PATH / "labels.csv"
SPLIT_FILE = DATA_PATH / "labels_with_splits.csv"

PATCH_FOLDER = DATA_PATH / "Patches"
CLASSMAP_FOLDER = DATA_PATH / "Classmap"
LABELS_FOLDER = DATA_PATH / "Labels"
HEC_FOLDER = DATA_PATH / "HEC"

SOURCE_PATCH_FOLDER = SOURCE_DATA / "Patches"
SOURCE_CLASSMAP_FOLDER = SOURCE_DATA / "Classmap"
SOURCE_LABELS_FOLDER = SOURCE_DATA / "Labels"
SOURCE_HEC_FOLDER = SOURCE_DATA / "HEC"

OUTPUT_DIR = Path(os.environ.get("CCRCC_LOGS", BASE_PATH / "logs"))
ML_RUNS = Path(os.environ.get("CCRCC_MLRUNS", BASE_PATH / "mlruns"))
MLFLOW_BACKUP_DIR = Path(
    os.environ.get("CCRCC_MLFLOW_BACKUP", BASE_PATH / "mlflow_backups")
)

MODEL_NAME = os.environ.get("CCRCC_MODEL", "google/vit-base-patch32-384")
