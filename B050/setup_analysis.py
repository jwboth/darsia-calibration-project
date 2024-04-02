"""Calibration of analysis object - detecting pH value."""

from fluidflower import FluidFlower
from data_management import data_manager
from pathlib import Path
from datetime import datetime
import numpy as np

# ! ---- PREPROCESSING ---- !

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

    # Pick sample calibration image
    calibration_path = list(sorted(data_manager["sample_folder"].glob("*.JPG")))[-1]

    # Define analysis object
    calibration_img = fluidflower.read_image(calibration_path)
    if True:
        # Calibration from scratch
        analysis = fluidflower.setup_analysis(calibration_img, width=5)
        characteristic_colors = analysis.characteristic_colors
        concentrations = analysis.concentrations
    elif False:
        # Calibration for given image and hAnd picked samples
        samples = [
            [(slice(145, 150, None), slice(4249, 4254, None))],
            [
                (slice(733, 738, None), slice(1414, 1419, None)),
                (slice(718, 723, None), slice(2032, 2037, None)),
                (slice(673, 678, None), slice(2666, 2671, None)),
                (slice(718, 723, None), slice(3902, 3907, None)),
            ],
            [
                (slice(2619, 2624, None), slice(1158, 1163, None)),
                (slice(2362, 2367, None), slice(1686, 1691, None)),
                (slice(2181, 2186, None), slice(2274, 2279, None)),
                (slice(2347, 2352, None), slice(2982, 2987, None)),
                (slice(2694, 2699, None), slice(4822, 4827, None)),
            ],
        ]
        concentrations = [[0.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0, 8.0]]
        zero_samples = [
            [],
            [(slice(764, 769, None), slice(449, 454, None))],
            [(slice(2754, 2759, None), slice(223, 228, None))],
        ]
        analysis = fluidflower.setup_analysis_from_samples(
            calibration_img, samples, concentrations, zero_samples
        )

    elif True:
        # Calibration based on predefined (relative) colors and concentrations
        # NOTE: It is independent of any calibration image.
        characteristic_colors = [
            [[-0.05846383, -0.07348965, -0.07187797], [0.0, 0.0, 0.0]],
            [
                [-0.20175312, -0.19924678, -0.63031232],
                [-0.29924101, -0.21978773, -0.53422451],
                [-0.44605044, -0.26609209, -0.48252413],
                [-0.6451894, -0.37212512, -0.34566534],
                [-0.01031403, 0.00577895, 0.0067595],
                [0.0, 0.0, 0.0],
            ],
            [
                [-0.41913643, -0.33610478, -0.44150516],
                [-0.58207703, -0.39748091, -0.437289],
                [-0.70064342, -0.45499918, -0.43600184],
                [-0.79348129, -0.53134435, -0.34465751],
                [-0.85962957, -0.58676749, -0.23366325],
                [-0.01547408, -0.00144339, 0.01524057],
                [0.0, 0.0, 0.0],
            ],
        ]
        concentrations = [
            [0.0, 0],
            [4.0, 5.0, 6.0, 7.0, 0, 0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 0, 0],
        ]
        analysis = fluidflower.setup_analysis_from_colors(
            characteristic_colors, concentrations
        )

    if True:
        date = datetime.now().strftime("%Y-%m-%d %H%M")
        results_folder = data_manager["results_folder"] / Path("calibration")
        results_folder.mkdir(parents=True, exist_ok=True)
        analysis_data = {
            "characteristic_colors": characteristic_colors,
            "concentrations": concentrations,
        }
        np.savez(
            results_folder / Path(f"analysis_{date}.npz"),
            analysis_data=analysis_data,
        )
        print(
            f"Saved calibration data to {results_folder / Path(f'analysis_{date}.npz')}"
        )

    # ! ---- ANALYZE IMAGES ---- !

    concentration = analysis(calibration_img)
    concentration.show()
