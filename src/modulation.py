# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from PIL import Image
from scipy.ndimage import gaussian_filter

from ccrcc_grading.config import SOURCE_LABELS_FOLDER, SOURCE_PATCH_FOLDER
from ccrcc_grading.modulation_configs import MODULATION_CONFIGS


class CCRCCPreprocessor:
    """
    Fuses a nuclei classification map into an RGB histopathology patch before it
    reaches the ViT, so no architectural change is needed to use semantic input.

    The classification map uses five labels: 0 background, 1-3 WHO/ISUP tumour
    grades, 4 non-tumorous cell. Label 4 is a cell-type label, not WHO/ISUP
    grade 4, and is deliberately given no enhancement.

    All methods here are multiplicative rather than additive. A multiplicative
    form scales the image gradient instead of replacing it, so RGB texture (the
    signal the ViT attention actually keys on) survives the fusion, which an
    additive overlay would partially destroy.
    """

    def __init__(self, method_configs):
        self.method = method_configs.get("method", "multiplicative")
        self.alpha = method_configs.get("alpha", 0.3)
        self.beta = method_configs.get("beta", 2.0)
        self.sigma_smooth = method_configs.get("sigma_smooth", 2.0)
        self.emphasis_grade = method_configs.get("emphasis_grade", 3)
        self.color_overlay_strength = method_configs.get("color_overlay_strength", 0.3)

        if method_configs.get("grade_weights", None) is None:
            self.grade_weights = {
                0: 1.00,  # background
                1: 1.10,
                2: 1.20,
                3: 1.35,  # grade 3 carries the most clinical weight
                4: 1.00,  # non-tumorous
            }
        else:
            self.grade_weights = method_configs["grade_weights"]

        # Colours chosen to be perceptually separable against H&E purple/pink.
        # Grade 3 gets green because it is furthest from the tissue background
        # and is the grade the model must not miss.
        self.grade_colors = {
            0: np.array([1.0, 1.0, 1.0]),
            1: np.array([0.85, 0.75, 0.95]),
            2: np.array([1.0, 1.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([1.0, 0.8, 0.8]),
        }

    def sigmoid_weight(self, grade: int) -> float:
        """
        Grade weighting f(c) = 1 / (1 + exp(-beta * (c - c0))).

        Grading is ordinal, not categorical, so the weight has to rise smoothly
        across grades rather than jump. beta sets the steepness; c0 sits half a
        grade below emphasis_grade so that the emphasised grade lands on the
        upper plateau of the sigmoid.
        """
        c0 = self.emphasis_grade - 0.5
        return 1.0 / (1.0 + np.exp(-self.beta * (grade - c0)))

    def create_modulation_map(
        self, class_map: np.ndarray, use_sigmoid: bool = False
    ) -> np.ndarray:
        """
        Turn a per-pixel class map into a per-pixel intensity multiplier.

        With use_sigmoid the base grade weight is scaled by the sigmoid, which
        keeps the ordinal spacing between grades; without it the weights are
        applied flat. Gaussian smoothing afterwards is what prevents nuclei
        boundaries from becoming hard edges that the ViT could latch onto as
        artefacts instead of morphology.
        """
        mod_map = np.ones_like(class_map, dtype=np.float32)

        if use_sigmoid:
            for grade in range(5):
                mask = class_map == grade
                weight = self.sigmoid_weight(grade)
                mod_map[mask] = 1.0 + (
                    weight * (self.grade_weights.get(grade, 1.0) - 1.0)
                )
        else:
            for grade, weight in self.grade_weights.items():
                mod_map[class_map == grade] = weight

        if self.sigma_smooth > 0:
            mod_map = gaussian_filter(mod_map, sigma=self.sigma_smooth)

        return mod_map

    def create_color_map(self, class_map: np.ndarray) -> np.ndarray:
        """Render the class map as a smoothed RGB overlay."""
        h, w = class_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.float32)

        for grade, color in self.grade_colors.items():
            mask = class_map == grade
            color_map[mask] = color

        if self.sigma_smooth > 0:
            for c in range(3):
                color_map[:, :, c] = gaussian_filter(
                    color_map[:, :, c], sigma=self.sigma_smooth
                )

        return color_map

    def color_emphasis(self, image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
        """
        Intensity modulation plus a colour overlay, blended 50/50.

        Intensity alone encodes grade as a scalar, which the model can confuse
        with staining variation. Adding a distinct hue per grade gives an
        explicitly categorical channel on top of the ordinal one. This is the
        configuration family that performed best; the overlay term is what
        separates it from plain multiplicative modulation.
        """
        img_float = image.astype(np.float32) / 255.0

        color_map = self.create_color_map(class_map)

        mod_map = self.create_modulation_map(class_map, use_sigmoid=True)
        mod_map_3d = np.stack([mod_map, mod_map, mod_map], axis=-1)

        # Multiplicative colour tint, centred so mid-grey leaves the pixel alone
        color_modulated = img_float * (1.0 + self.alpha * (color_map - 0.5))

        # Transparent colour filter, strength set by color_overlay_strength
        overlay_strength = self.color_overlay_strength
        color_overlay = (
            1.0 - overlay_strength
        ) * img_float + overlay_strength * color_map * img_float

        enhanced = 0.5 * color_modulated + 0.5 * color_overlay
        enhanced = enhanced * mod_map_3d

        return np.clip(enhanced * 255, 0, 255).astype(np.uint8)

    def multiplicative_modulation_strong(
        self, image: np.ndarray, class_map: np.ndarray
    ) -> np.ndarray:
        """
        Multiplicative modulation with the deviation from unity scaled by 1.5.

        Used to probe whether raw modulation strength alone can substitute for
        the colour overlay. It cannot: see the overlay-free rows in the results
        table.
        """
        mod_map = self.create_modulation_map(class_map, use_sigmoid=True)

        modulation = 1.0 + self.alpha * (mod_map - 1.0)
        modulation = 1.0 + (modulation - 1.0) * 1.5

        img_float = image.astype(np.float32)

        enhanced = np.zeros_like(img_float)
        for c in range(3):
            enhanced[:, :, c] = img_float[:, :, c] * modulation

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def multiplicative_modulation(
        self, image: np.ndarray, class_map: np.ndarray
    ) -> np.ndarray:
        """
        Baseline form: I'(x, y) = I(x, y) * (1 + alpha * f(C(x, y))).

        alpha controls how far the semantic map is allowed to move the image.
        As alpha approaches 0 the output converges on the untouched RGB patch,
        which makes it a clean ablation axis against the RGB-only baseline.
        """
        mod_map = self.create_modulation_map(class_map, use_sigmoid=True)

        modulation = 1.0 + self.alpha * (mod_map - 1.0)

        img_float = image.astype(np.float32)

        enhanced = np.zeros_like(img_float)
        for c in range(3):
            enhanced[:, :, c] = img_float[:, :, c] * modulation

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def additive_lab_enhancement(
        self, image: np.ndarray, class_map: np.ndarray
    ) -> np.ndarray:
        """
        Lightness enhancement in LAB rather than RGB.

        LAB is used because a fixed increment there is closer to a fixed
        perceptual step, so the grade hierarchy stays visually monotone instead
        of being distorted by the non-uniformity of RGB.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        attention_map = np.zeros_like(class_map, dtype=np.float32)
        attention_weights = {
            0: 0.0,
            1: 0.3,
            2: 0.8,
            3: 1.0,
            4: 0.0,
        }

        for grade, weight in attention_weights.items():
            attention_map[class_map == grade] = weight

        if self.sigma_smooth > 0:
            attention_map = gaussian_filter(attention_map, sigma=self.sigma_smooth)

        l_enhancement = attention_map * self.alpha * 30
        lab[:, :, 0] = lab[:, :, 0] + l_enhancement

        saturation_boost = 1.0 + (attention_map * self.alpha * 0.3)
        lab[:, :, 1] = (lab[:, :, 1] - 128) * saturation_boost + 128
        lab[:, :, 2] = (lab[:, :, 2] - 128) * saturation_boost + 128

        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)

        lab_uint8 = lab.astype(np.uint8)
        return cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)

    def edge_aware_enhancement(
        self, image: np.ndarray, class_map: np.ndarray
    ) -> np.ndarray:
        """
        Brighten nuclei boundaries, weighted by grade.

        Motivated by WHO/ISUP grading depending on nucleolar prominence, that
        is on boundary morphology rather than region area. Grade 3 edges get
        the strongest lift, grade 4 none.
        """
        enhanced = image.astype(np.float32)

        edge_strengths = {
            1: 0.1 * self.alpha,
            2: 0.35 * self.alpha,
            3: 0.5 * self.alpha,
            4: 0.0 * self.alpha,
        }

        for grade, strength in edge_strengths.items():
            mask = (class_map == grade).astype(np.uint8) * 255

            edges = cv2.Canny(mask, 50, 150)

            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            if self.sigma_smooth > 0:
                edges = gaussian_filter(
                    edges.astype(np.float32), sigma=self.sigma_smooth / 2
                )

            edge_enhancement = edges * strength
            for c in range(3):
                enhanced[:, :, c] += edge_enhancement

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def hybrid_enhancement(
        self, image: np.ndarray, class_map: np.ndarray
    ) -> np.ndarray:
        """Weighted mix of multiplicative and LAB enhancement, 0.6 / 0.4."""
        mult_enhanced = self.multiplicative_modulation(image, class_map)
        lab_enhanced = self.additive_lab_enhancement(image, class_map)

        mult_weight = 0.6
        lab_weight = 0.4

        hybrid = mult_weight * mult_enhanced.astype(
            np.float32
        ) + lab_weight * lab_enhanced.astype(np.float32)

        if self.alpha > 0.3:
            edge_enhanced = self.edge_aware_enhancement(image, class_map)
            hybrid = 0.9 * hybrid + 0.1 * edge_enhanced.astype(np.float32)

        return np.clip(hybrid, 0, 255).astype(np.uint8)

    def process_image(self, image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
        """Resize the class map to the image if needed, then dispatch on method."""
        if class_map.shape[:2] != image.shape[:2]:
            class_map = cv2.resize(
                class_map.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        method_map = {
            "multiplicative": self.multiplicative_modulation,
            "multiplicative_strong": self.multiplicative_modulation_strong,
            "color_emphasis": self.color_emphasis,
            "additive_lab": self.additive_lab_enhancement,
            "edge_aware": self.edge_aware_enhancement,
            "hybrid": self.hybrid_enhancement,
        }

        if self.method not in method_map:
            raise ValueError(f"Unknown method: {self.method}")

        return method_map[self.method](image, class_map)


def generate_config_name(method_configs):
    """
    Build a filesystem-safe name that fully identifies a configuration.

    Datasets are cached on disk by this name and MLflow runs are matched back
    to it later, so it has to encode every parameter that changes the pixels,
    including custom grade weights.
    """
    method = method_configs.get("method", "default")
    alpha = method_configs.get("alpha", 0.3)
    beta = method_configs.get("beta", 2.0)
    sigma = method_configs.get("sigma_smooth", 2.0)
    overlay = method_configs.get("color_overlay_strength", 0.3)

    method_short = {
        "multiplicative": "Mult",
        "multiplicative_strong": "MuSt",
        "color_emphasis": "CoEm",
        "additive_lab": "AdLa",
        "edge_aware": "Edge",
        "hybrid": "Hybr",
    }

    method_name = method_short.get(method, method[:4])

    name_parts = [
        "Mix",
        method_name,
        f"a{alpha:.2f}".replace(".", ""),
        f"b{beta:.1f}".replace(".", ""),
        f"s{sigma:.1f}".replace(".", ""),
    ]

    if "color" in method:
        name_parts.append(f"o{overlay:.2f}".replace(".", ""))

    if method_configs.get("grade_weights") is not None:
        weights = method_configs["grade_weights"]
        weight_str = "w" + "".join(
            [f"{weights.get(i, 1.0):.2f}".replace(".", "") for i in range(5)]
        )
        name_parts.append(weight_str)

    return "_".join(name_parts)


IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")


def find_images(source_path):
    files = []
    for pattern in IMAGE_GLOBS:
        files.extend(source_path.glob(pattern))
    return sorted(files)


def make_modulated_image(processor, image_file, labels_folder):
    image = np.array(Image.open(image_file))

    mat_file = labels_folder / (image_file.stem + ".mat")
    if not mat_file.exists():
        print(f"Warning: no .mat file found for {image_file.name}, skipping")
        return None

    mat_data = scipy.io.loadmat(mat_file)
    class_map = mat_data["class_map"]
    return processor.process_image(image, class_map)


def make_modulated_dataset(method_configs, source_path, labels_folder, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    image_files = find_images(source_path)
    processor = CCRCCPreprocessor(method_configs=method_configs)

    processed_count = 0
    skipped_count = 0

    for image_file in image_files:
        modulated = make_modulated_image(processor, image_file, labels_folder)

        if modulated is None:
            skipped_count += 1
            continue

        Image.fromarray(modulated).save(output_path / image_file.name)
        processed_count += 1

    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    source_path = SOURCE_PATCH_FOLDER
    labels_folder = SOURCE_LABELS_FOLDER
    output_path = Path(os.environ.get("CCRCC_FIGURES", "./figures"))
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = find_images(source_path)
    if not image_files:
        raise SystemExit(f"No images found in {source_path}")

    test_image_file = image_files[0]

    images = [np.array(Image.open(test_image_file))]
    config_names = ["Original"]

    for config in MODULATION_CONFIGS:
        processor = CCRCCPreprocessor(method_configs=config)
        modulated = make_modulated_image(processor, test_image_file, labels_folder)

        if modulated is not None:
            images.append(modulated)
            config_names.append(generate_config_name(config))

    n_cols = 3
    n_rows = (len(images) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (img, name) in enumerate(zip(images, config_names)):
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=10)
        axes[idx].axis("off")

    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path / "mixer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comparison for {test_image_file.name} written to {output_path}")
