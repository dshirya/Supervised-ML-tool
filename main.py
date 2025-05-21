import os
import time

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

from core import folder, preprocess, report
from core.models import PLS_DA, SVM, PLS_DA_plot, XGBoost

# Use the SAF's features as the "y" vector
df_SAF = pd.read_csv("data/class.csv")
y = df_SAF["Structure"]
validation_csv_file = "data/validation.csv"

# Initialize and fit the LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)
output_dir_path = os.path.join(script_dir_path, "features")

# Find all .csv files in folders
csv_file_paths = folder.find_csv_files(output_dir_path)

for i, csv_file_path in enumerate(csv_file_paths, start=1):
    start_time = time.perf_counter()

    # Load and preprocess the dataset
    X_df, X, columns = preprocess.prepare_standarlize_X_block_(csv_file_path)
    print(
        f"\nProcessing {csv_file_path} with {X.shape[1]} features ({i}/{len(csv_file_paths)})."
    )
    X_val_df, X_val = preprocess.preprocess_validation_data(
        validation_csv_file
    )

    print("(1/4) Running SVM model...")
    feature_file_name = folder.get_file_name(csv_file_path)
    SVM_model_report = SVM.get_report(X, y)
    report.record_model_performance(SVM_model_report, "SVM", csv_file_path)
    probabilities, y_pred_svm = SVM.validate_svc_with_probabilities(
        X, y, X_val, csv_file_path, validation_csv_file
    )

    print("(2/4) Running PLS_DA n=2...")
    PLS_DA_plot.plot_two_component(X, y, csv_file_path) 
    PLS_DA_plot.plot_two_component_with_validation(X, y, X_val, csv_file_path)

    print("(3/4) Running PLS_DA model with the best n...")
    file_name = folder.get_file_name(csv_file_path)
    # Determine the best number of components
    best_n_components = PLS_DA.find_best_n_dim(X, y_encoded, csv_file_path)
    best_pls = PLSRegression(n_components=best_n_components)
    PLA_DA_model_report = PLS_DA.generate_classification_report(X, y, best_pls)
    report.record_model_performance(PLA_DA_model_report, "PLS_DA", csv_file_path)
    PLS_DA.save_feature_importance(
        X, columns, y_encoded, best_pls, best_n_components, csv_file_path
    )
    # validation
    y_pred_val = PLS_DA.validate_PLS_DA_model(
        best_pls, X, y, X_val, csv_file_path, validation_csv_file
    )

    print("(4/4) Running XGBoost model...")
    XGBoost_model_report = XGBoost.run_XGBoost(X_df, y)
    report.record_model_performance(XGBoost_model_report, "XGBoost", csv_file_path)
    XGBoost.plot_XGBoost_feature_importance(X_df, y_encoded, csv_file_path)
    
    # validation
    y_pred_val_xgb = XGBoost.validate_XGBoost(
        X, y, X_val, csv_file_path, validation_csv_file
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"===========Elapsed time: {elapsed_time:0.2f} seconds===========")