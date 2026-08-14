# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

"""
Check that a perturbed class map was corrupted by the amount that was asked for.

The requested percentage applies to nuclei instances, not pixels, so the two
figures printed here will not match and are not supposed to. The instance
figure is the one that should track the requested level; the pixel figure just
reflects how large the corrupted nuclei happened to be.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def compare(modified_path, original_path, file_name, expected_percentage):
    modified_mat = sio.loadmat(str(Path(modified_path) / file_name))
    original_mat = sio.loadmat(str(Path(original_path) / file_name))

    class_map = modified_mat["class_map"]
    original_class_map = original_mat["class_map"]
    instance_map = modified_mat["instance_map"]

    unique_instances = np.unique(instance_map)
    unique_instances = unique_instances[unique_instances > 0]
    total_segments = len(unique_instances)

    modified_segments = 0
    segment_details = []

    for segment_id in unique_instances:
        segment_mask = instance_map == segment_id

        original_class = int(
            np.bincount(original_class_map[segment_mask].flatten()).argmax()
        )
        modified_class = int(np.bincount(class_map[segment_mask].flatten()).argmax())

        if original_class != modified_class:
            modified_segments += 1
            segment_details.append(
                {
                    "segment_id": segment_id,
                    "original_class": original_class,
                    "modified_class": modified_class,
                    "num_pixels": int(np.sum(segment_mask)),
                }
            )

    total_pixels = class_map.size
    different_pixels = int(np.sum(class_map != original_class_map))

    print(f"Segments: {total_segments}, modified: {modified_segments}")
    print(f"Requested modification: {expected_percentage}%")
    print(f"Actual, by segment: {100 * modified_segments / total_segments:.2f}%")
    print(f"Actual, by pixel:   {100 * different_pixels / total_pixels:.2f}%")

    print("First 10 modified segments:")
    for detail in segment_details[:10]:
        print(
            f"  segment {detail['segment_id']}: "
            f"{detail['original_class']} -> {detail['modified_class']} "
            f"({detail['num_pixels']} pixels)"
        )

    return class_map, original_class_map


def plot_comparison(class_map, original_class_map, output_path):
    diff_mask = class_map != original_class_map
    difference = class_map.astype(int) - original_class_map.astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    im1 = axes[0, 0].imshow(class_map, cmap="tab10", vmin=0, vmax=4)
    axes[0, 0].set_title("Modified Class Map")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(original_class_map, cmap="tab10", vmin=0, vmax=4)
    axes[0, 1].set_title("Original Class Map")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(difference, cmap="RdBu", vmin=-4, vmax=4)
    axes[1, 0].set_title("Difference (Modified - Original)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(diff_mask, cmap="gray")
    axes[1, 1].set_title(f"Different Pixels ({int(np.sum(diff_mask))} pixels)")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Comparison plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modified", required=True, help="Perturbed .mat directory")
    parser.add_argument("--original", required=True, help="Ground truth .mat directory")
    parser.add_argument("--file", required=True, help="File name, including .mat")
    parser.add_argument("--expected", type=float, default=80.0)
    parser.add_argument("--out", default="classmap_comparison.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    class_map, original_class_map = compare(
        args.modified, args.original, args.file, args.expected
    )
    plot_comparison(class_map, original_class_map, Path(args.out))
