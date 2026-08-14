# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

import copy
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# Which labels a nucleus of a given class is allowed to be corrupted into.
# Two constraints are encoded here at once. Every class can fall to 0, which is
# a nullified nucleus and stands in for a segmentation false negative. Grades 1
# and 3 cannot swap directly, because real nuclei classifiers confuse adjacent
# grades far more often than they skip one; letting 1 and 3 exchange would make
# the perturbation harsher than any observed model error profile.
SMART_TRANSITIONS = {
    0: [1, 2, 3, 4],
    1: [0, 2, 4],
    2: [0, 1, 3, 4],
    3: [0, 2, 4],
    4: [0, 1, 2, 3],
}

ALL_CLASSES = [0, 1, 2, 3, 4]


def introduce_classification_errors(
    error_percentage,
    csv_path,
    mat_folder_path,
    output_path,
    use_smart_transitions=True,
    random_seed=None,
):
    """
    Corrupt a fixed share of nuclei in each class map and write new .mat files.

    Errors are applied per nucleus instance, not per pixel. A pixel-level
    perturbation would speckle single nuclei with several grades, which no
    classifier produces; corrupting whole connected instances reproduces the
    failure mode of a real segmentation-plus-classification pipeline, where a
    nucleus is found correctly and then assigned the wrong grade outright.

    The dominant class within an instance is used as its current label, since
    the stored class map is per pixel and can disagree at instance borders.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    mat_folder_path = Path(mat_folder_path)
    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_names = pd.read_csv(csv_path)["Name"].tolist()
    saved_count = 0

    for image_name in image_names:
        base_name = Path(image_name).stem
        mat_file_path = mat_folder_path / f"{base_name}.mat"

        if not mat_file_path.exists():
            continue

        mat_copy = copy.deepcopy(sio.loadmat(str(mat_file_path)))

        classmap = mat_copy["class_map"]
        instance_map = mat_copy["instance_map"]

        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances > 0]
        num_segments = len(unique_instances)

        if num_segments > 0:
            num_errors = max(1, int(np.round(num_segments * error_percentage / 100)))
            segments_to_modify = random.sample(
                list(unique_instances), min(num_errors, num_segments)
            )

            for segment_id in segments_to_modify:
                segment_mask = instance_map == segment_id
                current_class = int(
                    np.bincount(classmap[segment_mask].flatten()).argmax()
                )

                if use_smart_transitions:
                    possible_classes = SMART_TRANSITIONS.get(current_class, ALL_CLASSES)
                else:
                    possible_classes = ALL_CLASSES

                possible_classes = [c for c in possible_classes if c != current_class]

                if possible_classes:
                    classmap[segment_mask] = random.choice(possible_classes)

        output_file = output_folder / f"{base_name}.mat"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sio.savemat(str(output_file), mat_copy)

        saved_count += 1

    print(f"Saved {saved_count} modified .mat files to {output_folder}")


if __name__ == "__main__":
    from ccrcc_grading.config import SOURCE_LABELS_FOLDER, SPLIT_FILE

    introduce_classification_errors(
        error_percentage=100,
        csv_path=Path(os.environ.get("CCRCC_SPLIT_FILE", SPLIT_FILE)),
        mat_folder_path=SOURCE_LABELS_FOLDER,
        output_path=Path(os.environ.get("CCRCC_PERTURBED_LABELS", "./modified_mat")),
        use_smart_transitions=True,
        random_seed=42,
    )
