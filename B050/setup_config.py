"""Setup of preprocessing.

This file is supposed to be a template for the BSc/DarSIA calibration project.

"""

# ! ---- IMPORTS ---- !

import json
from datetime import datetime
from pathlib import Path
from data_management import data_manager

import darsia

# ! ---- DATA MANAGEMENT ---- !

baseline_path = data_manager["baseline_path"]

# ! ---- UNMODIFIED BASELINE IMAGE ---- !
original_baseline = darsia.imread(baseline_path)

# ! ---- CORRECTION MANAGEMENT ---- !

# Idea: Apply three corrections:
# 1. Drift correction aligning images by simple translation with respect to the color checker.
# 2. Color correction applying uniform colors in the color checker.
# 3. Curvature correction to crop images to the right rectangular format.
# The order has to be applied in later scripts as well.

# Start with empty config file.
config = {}

# 1. Drift correction: Define ROI corresponding to colorchecker
# First define the config for the drift correction. This is done by selecting the ROI of the
# e.g. colorchecker. The ROI is defined by the four corners of the colorchecker.
# Later use: drift_correction = darsia.DriftCorrection(baseline_darsia_image, **config["drift"])
if False:
    config["drift"] = {}
    if True:
        # Use automatic color checker detection
        _, voxels = darsia.find_colorchecker(original_baseline, "upper_right")
        config["drift"]["roi"] = voxels
    elif False:
        # Use manual color checter detection - start at the mark close to the brown patch
        point_selector = darsia.PointSelectionAssistant(original_baseline)
        voxels = point_selector()
        config["drift"]["roi"] = voxels
    # Before storing, need to make dict json serializable
    config["drift"]["roi"] = config["drift"]["roi"].tolist()

# 2. Color correction: Mark the four corners of the color checker
# Later use: color_correction = darsia.ColorCorrection(**config["color"])
if False:
    config["color"] = {}
    if True:
        # Employ information from drift correction
        config["color"]["roi"] = config["drift"]["roi"]
    elif False:
        # Use automatic color checker detection
        _, voxels = darsia.find_colorchecker(original_baseline, "upper_right")
        config["color"]["roi"] = voxels
    elif False:
        # Use manual color checter detection - start at the mark close to the brown patch
        point_selector = darsia.PointSelectionAssistant(original_baseline)
        voxels = point_selector()
        config["color"]["roi"] = voxels
    config["color"]["roi"] = config["color"]["roi"].tolist()

# 3. Curvature correction: Crop image mainly. It is based on the unmodifed baseline image. All
# later images are assumed to be aligned with that one. Bulge effects are neglected.
crop_assistant = darsia.CropAssistant(original_baseline)
config["curvature"] = crop_assistant()
# Later use: curvature_correction = darsia.CurvatureCorrection(config=config["crop"])

# ! ---- STORE CONFIG TO FILE ---- !

# Store config to file and use current datetime as ID
date = datetime.now().strftime("%Y-%m-%d %H%M")
results_folder = data_manager["results_folder"] / Path("config")
results_folder.mkdir(exist_ok=True)
with open(results_folder / Path(f"preprocessing_{date}.json"), "w") as output:
    json.dump(config, output)
print(f"File saved to {results_folder / Path(f'preprocessing_{date}.json')}.")
