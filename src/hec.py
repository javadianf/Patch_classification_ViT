# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

"""
Classification map channel concatenation (HEC).

Colour deconvolution splits an H&E patch into hematoxylin and eosin channels.
The third channel, which carries almost no independent stain information, is
replaced by the nuclei classification map. The result is a three-channel image
that a standard pretrained ViT accepts unmodified, with semantic content in
place of a redundant colour channel rather than alongside it.
"""

import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image

# Fixed per-class colours, used when the class map is rendered for inspection
# or as the standalone Classmap data form.
CLASS_COLORS = {
    0: [255, 255, 255],  # background
    1: [0, 255, 0],  # grade 1
    2: [255, 0, 255],  # grade 2
    3: [255, 0, 0],  # grade 3
    4: [0, 255, 255],  # non-tumorous cell
}


def class_map_to_rgb(class_map, color_dict=CLASS_COLORS):
    h, w = class_map.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_dict.items():
        rgb_image[class_map == class_id] = color

    return rgb_image


def process_all_labels(labels_dir, output_dir):
    """Render every .mat class map in labels_dir as a colour PNG."""
    os.makedirs(output_dir, exist_ok=True)

    mat_files = glob.glob(os.path.join(labels_dir, "*.mat"))
    print(f"Found {len(mat_files)} .mat files")

    for mat_file in mat_files:
        base_name = os.path.splitext(os.path.basename(mat_file))[0]

        mat = sio.loadmat(mat_file)
        rgb_image = class_map_to_rgb(mat["class_map"])

        Image.fromarray(rgb_image).save(os.path.join(output_dir, f"{base_name}.png"))

    print(f"Saved {len(mat_files)} images to {output_dir}")


def load_image(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_od(I):
    return -np.log10((I.astype(np.float64) / 255) + 0.001)


def od_to_absorbance(od):
    return np.log10(1 / (10 ** (-od) + 0.001))


def od_to_transmittance(od):
    return 10 ** (-od)


def get_heb_matrix():
    """
    Stain vectors for hematoxylin, eosin and their orthogonal complement.

    H and E are the published Ruifrok and Johnston reference vectors. The third
    column is their cross product rather than a measured stain, which is
    exactly why it can be discarded and reused for the class map with no loss
    of stain information.
    """
    H = np.array([0.644211, 0.716556, 0.266844])
    E = np.array([0.092789, 0.954111, 0.283111])
    H /= np.linalg.norm(H)
    E /= np.linalg.norm(E)
    B = np.cross(H, E)
    B /= np.linalg.norm(B)
    return np.array([H, E, B]).T


def normalize_array(in_array):
    return np.uint8(cv2.normalize(in_array, None, 0, 255, cv2.NORM_MINMAX))


def get_he_channels(img, HEB_matrix, mode="mx1"):
    """
    Unmix an RGB patch into hematoxylin and eosin channels.

    H is taken in absorbance and E in transmittance. Nuclei read as high values
    in H, while E as transmittance keeps cytoplasm bright, so the two channels
    stay visually complementary once stacked.
    """
    if mode != "mx1":
        raise ValueError(f"Unsupported mode: {mode}")

    img_od = rgb_to_od(img)
    img_od_reshaped = img_od.reshape(-1, 3)
    unmixed = np.dot(img_od_reshaped, np.linalg.pinv(HEB_matrix.T)).reshape(
        img.shape[0], img.shape[1], -1
    )

    h_raw = od_to_absorbance(unmixed[:, :, 0])
    e_raw = od_to_transmittance(unmixed[:, :, 1])

    return normalize_array(h_raw), normalize_array(e_raw)


def consistent_map_grayscale(arr):
    """
    Map class labels onto evenly spaced grey levels 0, 64, 128, 192, 255.

    Even spacing keeps the ordinal distance between grades intact after the
    ViT normalises the channel, which raw labels 0-4 would not survive.
    """
    value_map = {0: 0, 1: 64, 2: 128, 3: 192, 4: 255}
    return np.vectorize(value_map.get)(arr).astype(np.uint8)


def consistent_map_rgb(arr):
    return class_map_to_rgb(arr)


def get_c_channel(file_path, mode="grayscale"):
    mat_contents = sio.loadmat(file_path)
    class_map = mat_contents["class_map"]

    if mode == "grayscale":
        return consistent_map_grayscale(class_map)
    if mode == "rgb":
        return consistent_map_rgb(class_map)
    raise ValueError(f"Unsupported mode: {mode}")


def create_hec_image(rgb_path, mat_path, HEB_matrix, c_mode="grayscale"):
    img = load_image(rgb_path)
    h_norm, e_norm = get_he_channels(img, HEB_matrix)
    c_norm = get_c_channel(mat_path, mode=c_mode)

    return {"H": h_norm, "E": e_norm, "C": c_norm}


def stack_hec(hec_img, order="HEC"):
    if order == "HEC":
        return np.stack((hec_img["H"], hec_img["E"], hec_img["C"]), axis=-1)
    if order == "CEH":
        return np.stack((hec_img["C"], hec_img["E"], hec_img["H"]), axis=-1)
    if order == "CHE":
        return np.stack((hec_img["C"], hec_img["H"], hec_img["E"]), axis=-1)
    raise ValueError(f"Unsupported channel order: {order}")


def process_all_images(
    rgb_folder, mat_folder, output_folder, c_mode="grayscale", order="HEC"
):
    os.makedirs(output_folder, exist_ok=True)

    HEB_matrix = get_heb_matrix()
    rgb_files = glob.glob(os.path.join(rgb_folder, "*.png"))
    print(f"Found {len(rgb_files)} RGB images")

    for rgb_path in rgb_files:
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        mat_path = os.path.join(mat_folder, f"{base_name}.mat")

        if not os.path.exists(mat_path):
            print(f"Warning: no .mat file for {base_name}, skipping")
            continue

        hec_img = create_hec_image(rgb_path, mat_path, HEB_matrix, c_mode=c_mode)
        hec_stacked = stack_hec(hec_img, order=order)

        output_path = os.path.join(output_folder, f"{base_name}.png")
        cv2.imwrite(output_path, cv2.cvtColor(hec_stacked, cv2.COLOR_RGB2BGR))

    print(f"Saved HEC images to {output_folder}")


if __name__ == "__main__":
    from ccrcc_grading.config import SOURCE_DATA

    process_all_images(
        SOURCE_DATA / "Patches",
        SOURCE_DATA / "Labels",
        SOURCE_DATA / "HEC",
        c_mode="grayscale",
        order="HEC",
    )
