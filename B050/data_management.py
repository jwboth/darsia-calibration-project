"""Define data management object for project B050."""

from pathlib import Path

# Folders
data_folder = Path("C:\\Users\\jbo062\\data\\resfyslocal\\FluidFlower\\B050")
sample_folder = data_folder / Path("sample")
results_folder = data_folder / Path("darsia")

# Define single baseline image
baseline_path = list(sorted(sample_folder.glob("*.JPG")))[0]

# Config path
config_path = (
    results_folder / Path("config") / Path("preprocessing_2024-03-31 2255.json")
)

# Manually segmented image
manual_segmentation_path = (
    data_folder / Path("manual_segmentation") / Path("segmented_DSC07100.png")
)

# Labels
segmentation_path = (
    results_folder / Path("segmentation") / Path("labels_2024-04-01 0131.npz")
)

# Porosity
porosity_path = results_folder / Path("porosity") / Path("porosity_2024-04-01 0234.npz")

# Calibration data for analysis
calibration_path = (
    results_folder / Path("calibration") / Path("analysis_2024-04-01 0657.npz")
)

# Store above paths in data_manager
data_manager = {
    "data_folder": data_folder,
    "sample_folder": sample_folder,
    "results_folder": results_folder,
    "manual_segmentation_path": manual_segmentation_path,
    "baseline_path": baseline_path,
    "config_path": config_path,
    "segmentation_path": segmentation_path,
    "porosity_path": porosity_path,
    "calibration_path": calibration_path,
}
