import os
from os import makedirs
from os.path import dirname, join

from core import folder


def find_csv_files(script_dir_path):
    csv_files = []
    skip_dirs = ["svm", "pls", "xgboost"]  # Directories to skip

    for root, dirs, files in os.walk(script_dir_path, topdown=True):
        # Modify dirs in-place to skip certain directories
        dirs[:] = [
            d for d in dirs if not any(skip_dir in d.lower() for skip_dir in skip_dirs)
        ]

        for file in files:
            # Check if the file ends with '.csv' and does not contain 'report' in its name
            if file.endswith(".csv") and "report" not in file.lower():
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_file_name(file_path):
    return file_path.split("/")[-1].split(".")[0]


def create_folder_get_output_path(
    model_name, csv_file_path, suffix="report", ext="csv"
):
    # Featurizer method directory CAF, SAF, CBFV, etc.
    featurizer_dir_path = dirname(csv_file_path)

    # Ex) binary_features
    feature_file_name = folder.get_file_name(csv_file_path)

    # Create the output directory
    output_dir = join(featurizer_dir_path, model_name, feature_file_name)
    makedirs(output_dir, exist_ok=True)
    output_path = join(output_dir, feature_file_name + f"_{suffix}.{ext}")

    return output_path
