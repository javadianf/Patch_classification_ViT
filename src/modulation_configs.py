# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

# Configurations swept for the preprocessing comparison. Each entry is one row
# of the results table. Grade weights follow the clinical hierarchy: background
# and non-tumorous cells fixed at 1.0, grade 3 always the largest.
#
# Order matters. The first three are the configurations carried into the
# sensitivity analysis, and the last is the low-alpha reference that stays
# closest to untouched RGB.

MODULATION_CONFIGS = [
    # Strongest overlay. Best balanced accuracy in the reported results.
    {
        "method": "color_emphasis",
        "alpha": 0.85,
        "beta": 3.0,
        "sigma_smooth": 1.5,
        "color_overlay_strength": 0.5,
        "grade_weights": {0: 1.0, 1: 1.25, 2: 1.60, 3: 2.00, 4: 1.00},
    },
    # Wider spacing between adjacent grades, weaker overlay.
    {
        "method": "color_emphasis",
        "alpha": 0.65,
        "beta": 2.5,
        "sigma_smooth": 2.0,
        "color_overlay_strength": 0.3,
        "grade_weights": {0: 1.0, 1: 1.25, 2: 1.55, 3: 1.85, 4: 1.05},
    },
    {
        "method": "color_emphasis",
        "alpha": 0.7,
        "beta": 2.5,
        "sigma_smooth": 2.0,
        "color_overlay_strength": 0.35,
        "grade_weights": {0: 1.0, 1: 1.20, 2: 1.50, 3: 1.85, 4: 1.00},
    },
    # sigma_smooth 0 isolates the effect of boundary smoothing.
    {
        "method": "color_emphasis",
        "alpha": 0.7,
        "beta": 3.0,
        "sigma_smooth": 0.0,
        "color_overlay_strength": 0.4,
        "grade_weights": {0: 1.0, 1: 1.20, 2: 1.60, 3: 2.00, 4: 1.00},
    },
    {
        "method": "color_emphasis",
        "alpha": 0.5,
        "beta": 2.0,
        "sigma_smooth": 2.0,
        "color_overlay_strength": 0.2,
        "grade_weights": {0: 1.0, 1: 1.15, 2: 1.40, 3: 1.70, 4: 1.00},
    },
    # No colour overlay. Tests whether modulation strength alone is enough.
    {
        "method": "multiplicative_strong",
        "alpha": 0.8,
        "beta": 4.0,
        "sigma_smooth": 2.0,
        "grade_weights": {0: 1.0, 1: 1.30, 2: 1.70, 3: 2.20, 4: 1.00},
    },
    # No overlay, sigmoid pushed as steep as swept, grade 3 weight at 2.5.
    {
        "method": "multiplicative_strong",
        "alpha": 0.9,
        "beta": 5.0,
        "sigma_smooth": 1.5,
        "grade_weights": {0: 1.0, 1: 1.15, 2: 1.50, 3: 2.50, 4: 1.00},
        "emphasis_grade": 3,
    },
    # Grades 2 and 3 weighted close together.
    {
        "method": "color_emphasis",
        "alpha": 0.75,
        "beta": 3.0,
        "sigma_smooth": 2.0,
        "color_overlay_strength": 0.4,
        "grade_weights": {0: 1.0, 1: 1.10, 2: 1.80, 3: 2.00, 4: 1.00},
    },
    # Low-alpha reference, near the RGB-only baseline.
    {
        "method": "multiplicative",
        "alpha": 0.3,
        "beta": 2.0,
        "sigma_smooth": 2.0,
        "grade_weights": {0: 1.0, 1: 1.10, 2: 1.20, 3: 1.35, 4: 1.00},
    },
]
