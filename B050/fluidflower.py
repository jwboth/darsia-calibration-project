"""Analysis object for the FluidFlower rig."""

import numpy as np
from pathlib import Path
import darsia
import matplotlib.pyplot as plt
import skimage
from typing import Optional
import json
from concentration_analysis import HeterogeneousMultichromaticCO2Analysis


class FluidFlower:
    def __init__(self, config_path: Path, baseline_path: Path):
        """Analysis object for the FluidFlower rig.

        Args:
            config_path (Path): path to the configuration file
            baseline_path (Path): path to the baseline image

        """

        # ! ---- BASELINE ----

        # Initialize baseline
        baseline = darsia.imread(baseline_path)

        # Set reference time
        self.reference_date = baseline.date

        # ! ---- SPECS ----
        self.width = 0.92
        self.height = 0.55
        self.depth = 0.02
        self.porosity = 0.44

        # Define corresponding geometry
        shape_metadata = baseline.shape_metadata()
        self.geometry = darsia.ExtrudedPorousGeometry(
            porosity=self.porosity, depth=self.depth, **shape_metadata
        )

        # ! ---- CORRECTIONS ----

        # Read configuration file and convert voxels to true voxels
        config_file = open(config_path)
        config = json.load(config_file)
        config["curvature"]["crop"]["pts_src"] = darsia.make_voxel(
            config["curvature"]["crop"]["pts_src"]
        )

        # Define translation correction object based on color checker
        if False:
            config["drift"]["roi"] = darsia.make_voxel(config["drift"]["roi"])
            drift_correction = darsia.DriftCorrection(baseline, config=config["drift"])

        # NOTE: This is the entire config file as produced through the assistant.
        # Convert voxels to true voxels
        curvature_correction = darsia.CurvatureCorrection(config=config["curvature"])
        curvature_corrected_baseline = curvature_correction(baseline)

        #         # Determine illumination correction - idea base the samples all within a single sand represented over the entire rig.
        #         if False:
        #             assistant = darsia.BoxSelectionAssistant(
        #                 curvature_corrected_baseline, width=200
        #             )
        #             samples = assistant()
        #             print(samples)
        #             raise
        #             assert False, "copy samples"
        #         else:
        #             # Hand-picked samples all from lower sand layer
        #             samples = [
        #                 (slice(4029, 4229, None), slice(592, 792, None)),
        #                 (slice(4074, 4274, None), slice(6987, 7187, None)),
        #                 (slice(2113, 2313, None), slice(6549, 6749, None)),
        #                 (slice(4029, 4229, None), slice(3578, 3778, None)),
        #                 (slice(1646, 1846, None), slice(3925, 4125, None)),
        #                 (slice(1827, 2027, None), slice(472, 672, None)),
        #                 (slice(3109, 3309, None), slice(864, 1064, None)),
        #             ]
        #
        #         illumination_correction = darsia.IlluminationCorrection(
        #             curvature_corrected_baseline,
        #             samples,
        #             ref_sample=-1,
        #             filter=lambda x: skimage.filters.gaussian(x, sigma=200),
        #             colorspace="lab-scalar",  # or "hsl-scalar"
        #             interpolation="rbf",
        #             show_plot=False,
        #         )
        #         illumination_corrected_baseline = illumination_correction(
        #             curvature_corrected_baseline
        #         )

        if False:
            # Define color correction object - target here the same colors as in
            # the original image (modulo curvature correction)
            colorchecker, cc_aligned_voxels = darsia.find_colorchecker(
                curvature_corrected_baseline, "upper_right"
            )
            color_correction = darsia.ColorCorrection(
                base=colorchecker, config={"roi": cc_aligned_voxels}
            )
            #        color_corrected_baseline = color_correction(illumination_corrected_baseline)

        # Define workflow of corrections
        self.corrections = [
            # drift_correction,
            curvature_correction,
            #            illumination_correction,
            # color_correction,
        ]

        # Define reference baseline
        self.baseline = self.read_image(baseline_path)

        # Plot access for debugging purposes
        if False:
            plt.figure("original")
            plt.imshow(curvature_corrected_baseline.img)
            #            plt.figure("illumination")
            #            plt.imshow(illumination_corrected_baseline.img)
            #            plt.figure("color")
            #            plt.imshow(color_corrected_baseline.img)
            plt.figure("baseline")
            plt.imshow(self.baseline.img)
            plt.show()

    def set_segmentation(self, path: Path) -> None:
        """Set segmentation for the analysis object.

        Args:
            path (Path): path to the segmentation image

        """

        # Read segmentation image
        segmentation = darsia.imread(path)

        # Resize labels to baseline image
        segmentation = darsia.resize(
            segmentation, ref_image=self.baseline, interpolation="inter_nearest"
        )

        # Set segmentation
        self.segmentation = segmentation

    def set_porosity(self, path: Path) -> None:
        """Set porosity for the analysis object.

        Args:
            path (Path): path to the porosity image

        """

        # Read porosity image
        porosity = darsia.imread(path)

        # Resize labels to baseline image
        porosity = darsia.resize(
            porosity, ref_image=self.baseline, interpolation="inter_nearest"
        )

        # Set porosity
        self.porosity = porosity

        # Upscaling
        if False:
            # Porosity-based heterogeneous TVD - slow...
            self.tvd = darsia.TVD(
                method="heterogeneous bregman",
                omega=self.porosity.img,
                weight=10,
                solver=darsia.Jacobi(10),
                isotropic=True,
                max_num_iter=40,
                eps=1e-4,
                verbose=False,
            )
        else:
            # Fast but less accurate as no image-porosity taken into account
            self.tvd = darsia.CombinedModel(
                [
                    darsia.Resize(fx=0.2, fy=0.2, interpolation="inter_area"),
                    darsia.TVD(weight=0.1, method="chambolle"),
                ]
            )

    def read_image(self, path: Path, date=None) -> darsia.Image:
        """Read image from file and apply corrections.

        Args:
            path (Path): path to the image file
            date (datetime, optional): date of the image. Defaults to None.

        """
        # TODO take care of times.

        # Read image from file and apply corrections
        img = darsia.imread(
            path,
            transformations=self.corrections,
            # date=date,
            reference_date=self.reference_date,
        )

        return img

    def setup_analysis(
        self,
        img: darsia.Image,
        width: int = 25,
        num_clusters: int = 5,
        include_baseline: bool = True,
    ) -> darsia.ConcentrationAnalysis:
        """Setup analysis object.

        Args:
            img (Image): calibration image
            width (int, optional): width of the analysis object. Defaults to 25.
            num_clusters (int, optional): number of clusters for the analysis object. Defaults to 5.
            include_baseline (bool, optional): include baseline in the analysis object. Defaults to True.

        Returns:
            ConcentrationAnalysis: analysis object

        """

        # Setup analysis object
        analysis = HeterogeneousMultichromaticCO2Analysis(
            self.baseline, self.segmentation, relative=True, restoration=self.tvd
        )
        analysis.calibrate(
            img,
            width,
            num_clusters,
            include_baseline,
        )

        return analysis

    def setup_analysis_from_samples(
        self,
        img: darsia.Image,
        samples: list,
        concentrations: Optional[list] = None,
        zero_samples: Optional[list] = None,
    ) -> darsia.ConcentrationAnalysis:
        """Setup analysis object.

        Args:
            img (Image): calibration image
            samples (dict): samples for calibration
            concentrations (dict): concentrations for calibration
            zero_samples (dict): zero samples for calibration

        Returns:
            ConcentrationAnalysis: analysis object

        """

        # Setup analysis object
        analysis = HeterogeneousMultichromaticCO2Analysis(
            self.baseline, self.segmentation, relative=True, restoration=self.tvd
        )
        analysis.calibrate_with_samples(img, samples, concentrations, zero_samples)

        return analysis

    def setup_analysis_from_colors(
        self,
        colors: list,
        concentrations: list,
    ) -> darsia.ConcentrationAnalysis:
        """Setup analysis object.

        Args:
            colors (list): colors for calibration
            concentrations (list): concentrations for calibration

        Returns:
            ConcentrationAnalysis: analysis object

        """

        # Setup analysis object
        analysis = HeterogeneousMultichromaticCO2Analysis(
            self.baseline, self.segmentation, relative=True, restoration=self.tvd
        )
        analysis.calibrate_with_colors_and_concentrations(colors, concentrations)

        return analysis

    def load_analysis(self, path: Path, key: str) -> darsia.ConcentrationAnalysis:
        """Load analysis object.

        Args:
            path (Path): path to the analysis file
            key (str): key of the analysis object

        Returns:
            ConcentrationAnalysis: analysis object

        """

        # Load data (dictionary)
        calibration_data = np.load(path, allow_pickle=True)[key].item()
        colors = calibration_data["characteristic_colors"]
        concentrations = calibration_data["concentrations"]

        # Setup analysis object
        return self.setup_analysis_from_colors(colors, concentrations)
