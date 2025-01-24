import pandas as pd

from core import folder


def record_model_performance(model_report, model_name, feature_file_path):
    # Convert the dictionary to a DataFrame
    report_df = pd.DataFrame(model_report).transpose()

    output_path = folder.create_folder_get_output_path(
        model_name, feature_file_path, suffix="report", ext="csv"
    )
    # Save the DataFrame to a CSV file
    report_df.round(3).to_csv(output_path, index=True)
