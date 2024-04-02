"""Segmentation of geometry."""

from data_management import data_manager
import darsia
from fluidflower import FluidFlower
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

if __name__ == "__main__":

    # ! ---- DATA MANAGEMENT ---- !

    # Read baseline and calibration images - use first and last samples
    baseline_path = data_manager["baseline_path"]

    # Fetch config file
    config_path = data_manager["config_path"]

    # Results folder
    results_folder = data_manager["results_folder"] / Path("segmentation")

    # Fetch manually segmented image
    segmentation_path = data_manager["manual_segmentation_path"]

    # ! ---- RIG INITIALIZATION ---- !

    # Initialize the museums rig
    fluidflower = FluidFlower(config_path=config_path, baseline_path=baseline_path)

    # Baseline image
    baseline = fluidflower.baseline

    # Fetch curvature correction
    curvature_correction = fluidflower.corrections[1]

    # ! ---- SEGMENTATION ---- !

    # Manually segmented image
    pre_segmented_image = darsia.imread(segmentation_path)
    segmented_image = curvature_correction(pre_segmented_image)

    # Define geometric segmentation using assistant
    assistant = darsia.LabelsAssistant(background=segmented_image, verbosity=True)
    labels = assistant()

    if False:
        plt.figure("labels")
        plt.imshow(labels.img)
        plt.imshow(baseline.img, alpha=0.5)
        plt.show()

    # Save labels
    date = datetime.now().strftime("%Y-%m-%d %H%M")
    results_folder.mkdir(exist_ok=True)
    labels.save(results_folder / Path(f"labels_{date}.npz"))
