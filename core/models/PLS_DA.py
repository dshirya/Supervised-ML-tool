import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from core import folder


def find_best_n_dim(X, y, csv_file_path, MAX_N_COMPONENTS=10):
    best_accuracy = 0
    best_n_components = 2

    # Initialize a DataFrame to store the results
    results = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)

    for n in range(2, MAX_N_COMPONENTS + 1):
        pls = PLSRegression(n_components=n, scale=False)
        y_pred_continuous = cross_val_predict(pls, X, y, cv=skf)
        y_pred = np.round(y_pred_continuous).astype(int)
        y_pred = np.clip(y_pred, 0, 9)

        accuracy = accuracy_score(y, y_pred)

        # Append the results to the DataFrame
        results.append({"n_component": n, "accuracy": accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_components = n

    # Save the DataFrame to a CSV file
    output_path = folder.create_folder_get_output_path(
        "PLS_DA", csv_file_path, suffix="n_analysis", ext="csv"
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    return best_n_components


def save_feature_importance(
    X, X_columns, y_encoded, pls, best_n_components, csv_file_path
):
    # Fit the PLS model
    X_pls, _ = pls.fit_transform(X, y_encoded)

    # Calculate the variance explained by each component for X
    total_variance_X = np.var(X, axis=0).sum()
    explained_variance_X = [
        np.var(X_pls[:, i]) / total_variance_X for i in range(best_n_components)
    ]

    # Create column names with explained variance
    column_names = [
        f"Component_{i+1} ({explained_variance_X[i]:.3%})"
        for i in range(best_n_components)
    ]

    # Extract weights (importance) of each feature for each component
    df = pd.DataFrame(
        pls.x_weights_,
        columns=column_names,
        index=X_columns,  # Use actual column names from the DataFrame
    )

    # Round the values to three decimal places
    df = df.round(3)

    # Generate output path and save the dataframe to CSV
    output_path = folder.create_folder_get_output_path(
        "PLS_DA", csv_file_path, suffix="feature_importance", ext="csv"
    )
    df.to_csv(output_path)


def save_correlation_matrix(X, X_columns, csv_file_path):
    output_path = folder.create_folder_get_output_path(
        "PLS_DA",
        csv_file_path,
        "correlation_matrix",
        "csv",
    )
    # Compute the correlation matrix of the scaled features
    df = pd.DataFrame(X, columns=X_columns).corr()
    df.to_csv(output_path)


def generate_classification_report(X_scaled, y, pls):
    # Import necessary modules
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Set up 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)

    # Suppress specific warnings during cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        # Perform cross-validation and get continuous predictions
        y_scores = cross_val_predict(pls, X_scaled, y_encoded, cv=skf, method="predict")

        # Convert continuous predictions to nearest class labels
        y_pred = np.rint(y_scores).astype(int)
        # Clip predictions to ensure they fall within the valid range for y_encoded
        y_pred = np.clip(y_pred, 0, len(encoder.classes_) - 1)

        # Decode predicted labels back to original
        y_pred_decoded = encoder.inverse_transform(y_pred)

        # Generate and return classification report
        class_report = classification_report(
            y, y_pred_decoded, digits=3, output_dict=True
        )

    return class_report



def validate_PLS_DA_model(pls, X_val, csv_file_path, validation_csv_file):
    """
    Validates the PLS-DA model on a given validation set.
    
    Parameters:
        pls: Trained PLSRegression model.
        X_val: Validation feature matrix.
        csv_file_path: Path of the training data (for output organization).
        validation_csv_file: Path of the validation data.

    Returns:
        y_pred_validation: Predicted classes for the validation set.
    """
    # Predict class probabilities for the validation set
    y_scores_validation = pls.predict(X_val)

    # Convert continuous predictions to nearest class labels
    y_pred_validation = np.rint(y_scores_validation).astype(int)

    # Clip predictions to valid range of classes
    y_pred_validation = np.clip(y_pred_validation, 0, 9)  # Adjust range if needed

    # Save predictions to a CSV file
    output_path = folder.create_folder_get_output_path(
        "PLS_DA", validation_csv_file, suffix="validation_predictions", ext="csv"
    )  
    validation_results = pd.DataFrame(
        {
            "Validation Sample": np.arange(1, len(X_val) + 1),
            "Predicted Class": y_pred_validation.flatten(),  # Ensure proper formatting
        }
    )
    validation_results.to_csv(output_path, index=False)

    #print(f"Validation predictions saved to {output_path}")
    return y_pred_validation