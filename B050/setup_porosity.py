"""Helper to understand which labels can be grouped based on the color."""

from datetime import datetime
from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
from data_management import data_manager
from fluidflower import FluidFlower
from concentration_analysis import HeterogeneousMultichromaticCO2Analysis

if __name__ == "__main__":

    # ! ---- RIG INITIALIZATION ---- !

    # Fetch baseline path
    baseline_path = data_manager["baseline_path"]

    # Fetch config file
    config_path = data_manager["config_path"]

    # Initialize the museums rig
    fluidflower = FluidFlower(config_path=config_path, baseline_path=baseline_path)
    fluidflower.set_segmentation(data_manager["segmentation_path"])

    # Baseline image
    baseline = fluidflower.baseline

    # Clip values
    baseline.img = np.clip(baseline.img, 0, 1)

    # Collect characteristic colors for each label
    data = []
    one_data = []
    zero_data = []
    for mask, label in darsia.Masks(fluidflower.segmentation, return_label=True):

        num_clusters = 5
        data_dim = 3

        # TODO Integrate into extract characteristic data
        flat_image = np.reshape(baseline.img, (-1, data_dim))
        flat_mask = np.ravel(mask.img)
        flat_image = flat_image[flat_mask]
        pixels = np.float32(flat_image)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-2,
        )
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, num_clusters, None, criteria, 10, flags)

        # Determine most common color
        _, counts = np.unique(labels, return_counts=True)
        common_color = palette[np.argmax(counts)]
        least_common_color = palette[np.argmin(counts)]

        # Find the two points furthest away from each other.
        # Build a distance matrix.
        distance_matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(num_clusters):
                distance_matrix[i, j] = np.linalg.norm(palette[i] - palette[j])

        # Reduce to colors at least as bright as the common color
        for i in range(num_clusters):
            if np.linalg.norm(palette[i]) < np.linalg.norm(common_color):
                distance_matrix[i, :] *= 0.0
                distance_matrix[:, i] *= 0.0

        # Check if common color is the brightest
        if np.allclose(distance_matrix, 0):

            dark_color = common_color
            bright_color = least_common_color

        else:

            # Find max entry
            ind = np.unravel_index(
                np.argmax(distance_matrix, axis=None), distance_matrix.shape
            )
            i_max = ind[0]
            j_max = ind[1]

            # Determine the brighter and lighter color via norm
            if np.linalg.norm(palette[i_max]) > np.linalg.norm(palette[j_max]):
                bright_color = palette[i_max]
                dark_color = palette[j_max]
            else:
                bright_color = palette[j_max]
                dark_color = palette[i_max]

        one_data.append(dark_color)
        zero_data.append(bright_color)

        # Plot all data points to get an impression of the clustering - for debugging.
        if label in []:
            # print("palette")
            # print(palette)
            # print("bright")
            # print(bright_color)
            # print("dark")
            # print(dark_color)
            # print("common")
            # print(common_color)
            c = np.clip(np.abs(palette), 0, 1)
            plt.figure("Relative dominant colors")
            ax = plt.axes(projection="3d")
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")
            ax.scatter(
                palette[:, 0],
                palette[:, 1],
                palette[:, 2],
                c=c,
            )
            ax.scatter(
                bright_color[0], bright_color[1], bright_color[2], c="r", alpha=0.5
            )
            ax.scatter(dark_color[0], dark_color[1], dark_color[2], c="g", alpha=0.5)
            ax.scatter(
                common_color[0], common_color[1], common_color[2], c="b", alpha=0.5
            )
            plt.figure("img")
            plt.imshow(baseline.img)
            plt.imshow(mask.img, alpha=0.3)
            plt.figure("img origial")
            plt.imshow(baseline.img)
            plt.figure("labels")
            plt.imshow(fluidflower.segmentation.img)
            plt.show()

    # Extract image-porosity based on labels
    class PorosityAnalysis(HeterogeneousMultichromaticCO2Analysis): ...

    porosity_analysis = PorosityAnalysis(
        baseline,
        fluidflower.segmentation,
        relative=False,
        kernel=darsia.GaussianKernel(gamma=10),
    )
    porosity_analysis.calibrate_with_colors(one_data, zero_data)
    porosity = porosity_analysis(baseline)
    porosity_raw = porosity.copy()
    porosity.img = np.clip(porosity.img, 0, 1)

    # NOTE: This step is hardcoded - cut off values which are low.
    porosity.img[porosity.img < 0.4] = 0

    if True:
        plt.figure("labels")
        plt.imshow(fluidflower.segmentation.img)
        plt.figure("porosity")
        plt.imshow(porosity.img)
        plt.figure("raw")
        plt.imshow(porosity_raw.img)
        plt.figure("base")
        plt.imshow(baseline.img)
        plt.show()

    # Cache labels with current date
    if True:
        date = datetime.now().strftime("%Y-%m-%d %H%M")
        results_folder = data_manager["results_folder"] / Path("porosity")
        results_folder.mkdir(parents=True, exist_ok=True)
        porosity.save(results_folder / Path(f"porosity_{date}.npz"))
