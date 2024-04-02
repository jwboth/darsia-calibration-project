from typing import Optional

import darsia
import numpy as np


class HomogeneousCO2Detector(
    darsia.ConcentrationAnalysis,
):
    """Benchmark version for detecting 'any' CO2, yet without taking care of heterogeneous
    layering.

    """

    def __init__(
        self, base: darsia.Image
    ) -> None:  # , geometry: darsia.Geometry) -> None:
        """Contructor.

        Args:
            base (darsia.Image): baseline image

        """
        # Define a monochromatic signal converter based on the gray color.
        signal_reduction = darsia.MonochromaticReduction(color="negative-key")

        # Define an restoration routine. To speed up significantly the process,
        # invoke resizing of signals within the concentration analysis.
        restoration = darsia.CombinedModel(
            [
                darsia.Resize(fx=0.2, fy=0.2, interpolation="inter_area"),
                darsia.TVD(weight=0.1, method="chambolle"),
            ]
        )

        # Threshold low vaue
        model = darsia.StaticThresholdModel(
            0.25
        )  # 0.25 to strict for some layers...OK for lower left quadrant

        # Define general ConcentrationAnalysis
        config = {
            "diff option": "absolute",
            "restoration -> model": True,
        }
        super().__init__(
            base=base,
            signal_reduction=signal_reduction,
            restoration=restoration,
            model=model,
            **config,
        )


class MultichromaticConcentrationAnalysis(
    darsia.ConcentrationAnalysis, darsia.InjectionRateModelObjectiveMixin
):
    """Concentration analysis for CO2."""

    def __init__(
        self,
        baseline: darsia.Image,
        relative: bool = True,
        threshold: Optional[float] = None,
    ) -> None:
        """Contructor.

        Args:
            baseline (darsia.Image): baseline image
            relative (bool): flag controlling whether the concentration analysis is
                based image differences

        """
        # Cache
        self.baseline = baseline

        # Define an restoration routine. To speed up significantly the process,
        # invoke resizing of signals within the concentration analysis.
        restoration = darsia.CombinedModel(
            [
                darsia.Resize(fx=0.2, fy=0.2, interpolation="inter_area"),
                darsia.TVD(weight=0.2, method="chambolle"),
            ]
            + [
                darsia.StaticThresholdModel(threshold),
                darsia.TypeCorrection(np.float32),
                darsia.Resize(ref_image=self.baseline, interpolation="inter_nearest"),
                darsia.TypeCorrection(bool),
            ]
            if threshold is not None
            else [darsia.Resize(ref_image=self.baseline, interpolation="inter_area")]
        )

        # Start with non-calibrated model
        model = darsia.CombinedModel(
            [darsia.KernelInterpolation(darsia.GaussianKernel(gamma=9.73))]
        )

        # Define general ConcentrationAnalysis
        config = {
            "diff option": "plain",
            "restoration -> model": True,
        }
        super().__init__(
            base=baseline if relative else None,
            restoration=restoration,
            model=model,
            **config,
        )

    def calibrate(self, calibration_image) -> None:
        """
        Use last caliration image to define support points.

        Use all to fix the support points assignment.

        """
        # ! ---- STEP 0: Deactivate model and restoration

        model_cache = self.model
        restoration_cache = self.restoration
        self.model = None
        self.restoration = None

        # ! ---- STEP 1: Calibrate the support points (x) based on some images

        # Define characteristic points and corresponding data values
        if not hasattr(self, "samples"):
            print("Define samples")
            assistant = darsia.BoxSelectionAssistant(calibration_image, width=25)
            samples = assistant()
            print(samples)
            assert False, "cache samples"
        else:
            samples = self.samples
        if not hasattr(self, "zero_samples"):
            print("Define zero samples")
            assistant = darsia.BoxSelectionAssistant(calibration_image, width=25)
            zero_samples = assistant()
            print(zero_samples)
            assert False, "cache zero_samples"
        else:
            zero_samples = self.zero_samples
        concentrations = len(samples) * [1] + len(zero_samples) * [0]
        concentrations_base = len(samples) * [0]

        # Apply concentration analysis modulo the model and the restoration
        pre_concentration_base = self(self.baseline)
        pre_concentration = self(calibration_image)

        # Fetch characteristic colors from samples
        characteristic_colors_base = darsia.extract_characteristic_data(
            signal=pre_concentration_base.img, samples=samples, show_plot=False
        )
        characteristic_colors = darsia.extract_characteristic_data(
            signal=pre_concentration.img,
            samples=samples + zero_samples,
            show_plot=False,
        )

        # Collect data
        characteristic_colors = np.vstack(
            (characteristic_colors_base, characteristic_colors)
        )
        concentrations = np.array(concentrations_base + concentrations)

        # Update kernel interpolation
        model_cache[0].update(supports=characteristic_colors, values=concentrations)

        # Reinstall the model and the restoration
        self.restoration = darsia.CombinedModel(
            [
                model_cache,
                restoration_cache,
            ]
        )


class MultichromaticDetector(MultichromaticConcentrationAnalysis):
    def __init__(
        self, baseline: darsia.Image, threshold: float, relative: bool = True
    ) -> None:
        super().__init__(baseline, relative, threshold=threshold)


class HeterogeneousMultichromaticCO2Analysis(darsia.ConcentrationAnalysis):
    """Concentration analysis tailored to labeled media.

    Essentially, it offers merely a short-cut definition, accompanied by calibration.

    """

    def __init__(
        self,
        baseline: darsia.Image,
        labels: darsia.Image,
        relative: bool = True,
        show_plot: bool = False,
        use_tvd: bool = False,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            baseline (Image): baseline image; relevant for relative analysis as well as
                calibration in a comparative sense (on multiple images)
            labels (Image): labeled image
            relative (bool): flag controlling whether the analysis is relative
            show_plot (bool): flag controlling whether intermediate plots are showed
            use_tvd (bool): flag controlling whether TVD is an integral part
            kwargs: other keyword arguments

        """

        # Define an restoration routine. To speed up significantly the process,
        # invoke resizing of signals within the concentration analysis.
        if "restoration" in kwargs:
            restoration = darsia.CombinedModel([kwargs.get("restoration")])
        elif use_tvd:
            restoration = darsia.CombinedModel(
                [
                    # darsia.Resize(fx=0.2, fy=0.2, interpolation="inter_area"),
                    darsia.TVD(weight=0.2, method="chambolle", max_num_iter=1),
                    # darsia.TVD(weight=0.2, method="isotropic bregman"),
                ]
            )
        else:
            restoration = None

        # Define non-calibrated model in a heterogeneous fashion
        kernel = kwargs.get("kernel", darsia.GaussianKernel(gamma=1))
        model = darsia.CombinedModel(
            [
                darsia.HeterogeneousModel(
                    darsia.KernelInterpolation(kernel),
                    labels,
                )
            ]
        )

        # Define general config options
        config = {
            "diff option": "plain",
            "restoration -> model": False,
        }

        self.characteristic_colors = []
        """Characteristic colors for each label."""
        self.concentrations = []
        """Concentration values for each label."""

        # Define general ConcentrationAnalysis
        super().__init__(
            base=baseline if relative else None,
            restoration=restoration,
            labels=labels,
            model=model,
            **config,
        )

        # Cache meta information
        self.show_plot = show_plot

    def calibrate(
        self,
        calibration_image,
        width: int = 25,
        num_clusters: int = 5,
        include_baseline: bool = True,
    ) -> None:
        """
        Use last caliration image to define support points.

        Use all to fix the support points assignment.

        Args:
            calibration_image (Image): calibration image for extracting colors
            width (int): width of sample boxes returned from assistant - irrelevant if
                boxed defined
            num_clusters (int): number of characteristic clusters extracted
            include_baseline (bool): flag controlling whether baseline image is included

        """
        # ! ---- STEP 0: Deactivate model and restoration

        model_cache = self.model
        restoration_cache = self.restoration
        self.model = None
        self.restoration = None

        # ! ---- STEP 1: Calibrate the support points (x) based on some images

        # Check whether support points are provided or need setup
        setup_samples = not hasattr(self, "samples_collection") and not hasattr(
            self, "concentrations_collection"
        )
        setup_zero_samples = not hasattr(self, "zero_samples_collection")

        # Initialize data collections
        self.characteristic_colors = []
        self.concentrations = []

        for i, mask in enumerate(darsia.Masks(self.labels)):

            # Define characteristic points and corresponding data values
            if setup_samples:
                # Init
                if i == 0:
                    samples_collection = []
                    concentrations_collection = []

                # Setup
                print("Define samples")
                assistant = darsia.BoxSelectionAssistant(
                    calibration_image, background=mask, width=width
                )
                samples = assistant()
                print("Define associated concentration values - assumed 1 if empty")
                # Ask for concentration values from user
                concentrations = [
                    float(input(f"Concentration for sample {i}: "))
                    for i in range(len(samples))
                ]
                # concentrations = len(samples) * [1]

                # Cache
                samples_collection.append(samples)
                concentrations_collection.append(concentrations)
            else:
                samples = self.samples_collection[i]
                concentrations = self.concentrations_collection[i]
            if setup_zero_samples:
                # Init
                if i == 0:
                    zero_samples_collection = []

                # Setup
                print("Define zero samples")
                assistant = darsia.BoxSelectionAssistant(
                    calibration_image, background=mask, width=width
                )
                zero_samples = assistant()

                # Cache
                zero_samples_collection.append(zero_samples)
            else:
                zero_samples = self.zero_samples_collection[i]

            # Define data (1's and 0's)
            zero_concentrations = len(zero_samples) * [0]
            concentrations = concentrations + zero_concentrations

            # Apply concentration analysis modulo the model and the restoration
            pre_concentration = self(calibration_image)

            # Fetch characteristic colors from samples
            characteristic_colors = darsia.extract_characteristic_data(
                signal=pre_concentration.img,
                samples=samples + zero_samples,
                show_plot=self.show_plot,
                num_clusters=num_clusters,
            )

            # Use baseline image to collect 0-data
            if include_baseline:
                # Define zero data
                concentrations_base = len(samples) * [0]

                # Apply concentration analysis modulo the model and the restoration
                pre_concentration_base = self(self.base)

                # Fetch characteristic colors from samples
                characteristic_colors_base = darsia.extract_characteristic_data(
                    signal=pre_concentration_base.img,
                    samples=samples,
                    show_plot=self.show_plot,
                    num_clusters=num_clusters,
                )

                # Collect data
                characteristic_colors = np.vstack(
                    (characteristic_colors_base, characteristic_colors)
                )
                concentrations = np.array(concentrations_base + concentrations)

            # Update kernel interpolation
            model_cache[0][i].update(
                supports=characteristic_colors, values=concentrations
            )

            # Cache data
            self.characteristic_colors.append(characteristic_colors)
            self.concentrations.append(concentrations)

        # Reinstall the model and the restoration
        self.model = model_cache
        self.restoration = restoration_cache

        if setup_samples or setup_zero_samples:
            print("samples collection")
            print(samples_collection)
            print("concentrations collection")
            print(concentrations_collection)
            print("zero samples collection")
            print(zero_samples_collection)

    def calibrate_with_samples(
        self,
        calibration_image: darsia.Image,
        samples: list,
        concentrations: Optional[list] = None,
        zero_samples: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Calibrate analysis object.

        NOTE: Implicitly assumes a relative analysis.

        Assign unit concentrations to characteristic colors of all samples, while 0 gets
        mapped to 0.

        Args:
            calibration_image (Image): calibration image for extracting colors
            samples (list): sample data, providing samples for each label
            concentrations (list): concentration data, providing concentrations for each
                label
            zero_samples (list): zero sample data, providing zero samples for each label

        """

        # ! ---- STEP 0: Deactivate model and restoration

        model_cache = self.model
        restoration_cache = self.restoration
        self.model = None
        self.restoration = None

        # ! ---- STEP 1: Calibrate the support points (x) based on some images

        characteristic_colors_collection = []
        concentrations_collection = []

        for i in range(len(samples)):

            # Fetch samples
            sample = samples[i]
            zero_sample = zero_samples[i] if zero_samples is not None else []

            # Apply (relative) concentration analysis modulo the model and the restoration
            pre_concentration = self(calibration_image)

            # Fetch characteristic colors from samples
            characteristic_colors = darsia.extract_characteristic_data(
                signal=pre_concentration.img, samples=sample
            )

            # Define corresponding data (1's)
            concentration_values = (
                len(sample) * [1] if concentrations is None else concentrations[i]
            )

            # if "const" in kwargs:
            #    characteristic_colors = kwargs.get("const")
            #    concentration_values = len(characteristic_colors) * [1]

            # Append 0-data
            if len(zero_sample) > 0:
                zero_characteristic_colors = darsia.extract_characteristic_data(
                    signal=pre_concentration.img, samples=zero_sample
                )
                characteristic_colors = np.vstack(
                    (characteristic_colors, zero_characteristic_colors)
                )
                concentration_values = concentration_values + len(zero_sample) * [0]

            # Append zero data - assume relative analysis
            characteristic_colors = np.vstack(
                (characteristic_colors, np.zeros((1, 3), dtype=float))
            )
            concentration_values = concentration_values + [0]

            # Update kernel interpolation
            model_cache[0][i].update(
                supports=characteristic_colors, values=concentration_values
            )

            # Cache data
            characteristic_colors_collection.append(characteristic_colors)
            concentrations_collection.append(concentration_values)

        # Reinstall the model and the restoration
        self.model = model_cache
        self.restoration = restoration_cache

        if True:
            print("characteristic colors")
            print(characteristic_colors_collection)
            print("concentrations")
            print(concentrations_collection)

    def calibrate_with_colors(self, one_colors, zero_colors) -> None:
        """
        Use last caliration image to define support points.

        Use all to fix the support points assignment.

        Args:
            width (int): width of sample boxes returned from assistant - irrelevant if
                boxed defined
            num_clusters (int): number of characteristic clusters etracted in data an

        """
        # ! ---- STEP 1: Calibrate the support points (x) based on some images

        for i, mask in enumerate(darsia.Masks(self.labels)):

            # Define data (1's and 0's)
            concentrations = [
                1,
                0,
            ]  # one_colors[i].shape[0] * [1] + zero_colors[i].shape[0] * [0]
            colors = np.vstack((one_colors[i], zero_colors[i]))

            print(concentrations)
            print(colors)

            # Update kernel interpolation
            self.model[0][i].update(supports=colors, values=concentrations)

    def calibrate_with_colors_and_concentrations(self, colors, concentrations) -> None:
        """Calibrate analysis object from colors and concentrations.

        Args:
            colors (list): list of colors
            concentrations (list): list of concentrations

        """
        # ! ---- STEP 1: Calibrate the support points (x) based on some images

        for i, _ in enumerate(darsia.Masks(self.labels)):

            # Update kernel interpolation
            self.model[0][i].update(supports=colors[i], values=concentrations[i])
