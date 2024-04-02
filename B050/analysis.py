"""Example script for applying calibrated analysis to B050."""

from fluidflower import FluidFlower
from data_management import data_manager
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":

    # ! ---- RIG INITIALIZATION ---- !

    # Fetch baseline path
    baseline_path = data_manager["baseline_path"]

    # Fetch config file
    config_path = data_manager["config_path"]

    # Initialize the museums rig
    fluidflower = FluidFlower(config_path=config_path, baseline_path=baseline_path)
    fluidflower.set_segmentation(data_manager["segmentation_path"])
    fluidflower.set_porosity(data_manager["porosity_path"])

    # ! ---- CALIBRATE ANALYSIS ---- !

    # Load calibration data
    analysis = fluidflower.load_analysis(
        data_manager["calibration_path"], key="analysis_data"
    )

    # ! ---- ANALYZE IMAGES ---- !

    for path in list(sorted(data_manager["sample_folder"].glob("*.JPG"))):

        # Read image
        img = fluidflower.read_image(path)

        # Analyze image
        concentration = analysis(img)

        # Cut off negative values
        if True:
            concentration.img = np.clip(concentration.img, 0, None)

        # Some simple data processing
        print(f"Analyzing image {path.stem}.")
        print(f"Time of image: {img.time}.")
        print(f"Mean pH: {np.mean(concentration.img)}.")
        print(f"Integrated pH: {fluidflower.geometry.integrate(concentration)}.")
        print()

        # Display concentration and image
        if True:
            plt.figure("Image")
            plt.imshow(img.img)
            plt.figure("Concentration")
            plt.imshow(concentration.img)
            plt.colorbar()
            plt.show()

        # Save concentration in raw format - note high memory consumption
        if True:
            results_folder = data_manager["results_folder"] / Path("concentration")
            results_folder.mkdir(parents=True, exist_ok=True)
            concentration.save(results_folder / Path(f"{path.stem}.npz"))

        # Save concentration as image
        if True:
            results_folder = data_manager["results_folder"] / Path(
                "concentration_image"
            )
            results_folder.mkdir(parents=True, exist_ok=True)
            plt.imsave(
                results_folder / Path(f"{path.stem}.png"),
                concentration.img,
                cmap="viridis",
            )
