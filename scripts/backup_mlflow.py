# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

"""Copy the MLflow store to a backup location, and restore it back."""

import shutil
from datetime import datetime

from ccrcc_grading.config import MLFLOW_BACKUP_DIR, ML_RUNS

BACKUP_PREFIX = "mlruns_backup_"


def backup_mlruns():
    if not ML_RUNS.exists():
        print(f"Nothing to back up, {ML_RUNS} does not exist")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = MLFLOW_BACKUP_DIR / f"{BACKUP_PREFIX}{timestamp}"

    MLFLOW_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Backing up {ML_RUNS} to {destination}")
    shutil.copytree(ML_RUNS, destination)
    print("Backup complete")

    return destination


def list_backups():
    if not MLFLOW_BACKUP_DIR.exists():
        print("No backup directory found")
        return []

    backups = sorted(
        d
        for d in MLFLOW_BACKUP_DIR.iterdir()
        if d.is_dir() and d.name.startswith(BACKUP_PREFIX)
    )

    for idx, backup in enumerate(backups, 1):
        print(f"{idx}. {backup.name}")

    return backups


def recover_backup():
    backups = list_backups()
    if not backups:
        return

    while True:
        choice = input("Backup number to recover, or q to quit: ")
        if choice.lower() == "q":
            return
        try:
            index = int(choice)
        except ValueError:
            print("Enter a number")
            continue
        if 1 <= index <= len(backups):
            backup_path = backups[index - 1]
            break
        print(f"Enter a number between 1 and {len(backups)}")

    if ML_RUNS.exists():
        confirm = input(f"This will replace {ML_RUNS}. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            return
        shutil.rmtree(ML_RUNS)

    print(f"Recovering {backup_path} to {ML_RUNS}")
    shutil.copytree(backup_path, ML_RUNS)
    print("Recovery complete")


def main():
    actions = {"1": backup_mlruns, "2": list_backups, "3": recover_backup}

    while True:
        print("1. Create backup")
        print("2. List backups")
        print("3. Recover from backup")
        print("4. Exit")

        choice = input("Select an option (1-4): ")
        if choice == "4":
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid option")
            continue
        action()


if __name__ == "__main__":
    main()
