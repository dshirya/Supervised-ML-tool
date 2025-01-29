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


# Define the class-to-integer mapping
class_mapping = {'Cu3Au': 2, 'Cr3Si': 4, 'PuNi3': 6, 'Fe3C': 3, 'Mg3Cd': 1, 'TiAl3': 5} #based on Silhouettes scores

def encode_classes(y, class_mapping):
    """Encodes the class labels using the provided mapping."""
    return np.array([class_mapping[label] for label in y])

def decode_classes(y_encoded, class_mapping):
    """Decodes the integer class labels back to their string representation."""
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    return np.array([reverse_mapping[label] for label in y_encoded])

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
    y_encoded = encoder.fit_transform(y) + 1

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

def validate_PLS_DA_model(pls, X, y, X_val, csv_file_path, validation_csv_file):
    """
    Validates the PLS-DA model using one-vs-rest probabilities for classification.

    Parameters:
        pls: Trained PLSRegression model.
        X: Training feature matrix.
        y: Training target array.
        X_val: Validation feature matrix.
        csv_file_path: Path of the training data.
        validation_csv_file: Path of the validation data.

    Returns:
        validation_results: DataFrame with predicted class probabilities.
    """

    # Encode class labels using the predefined mapping
    y_encoded = encode_classes(y, class_mapping)

    # Convert y_encoded to one-hot encoding for regression
    y_one_hot = pd.get_dummies(y_encoded)

    # Train the PLS model
    pls.fit(X, y_one_hot)

    # Get raw predictions for the validation set
    y_scores_validation = pls.predict(X_val)

    # Ensure all predictions are positive and normalize them to probabilities (one-vs-rest)
    probabilities = np.maximum(y_scores_validation, 0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Assign predicted class based on the highest probability
    y_pred_validation_encoded = np.argmax(probabilities, axis=1) + 1  # Classes start from 1

    # Save results in a DataFrame
    column_names = [f"Class_{i}" for i in range(1, len(class_mapping) + 1)]
    validation_results = pd.DataFrame(probabilities, columns=column_names)
    validation_results["Predicted Class"] = y_pred_validation_encoded
    validation_results.insert(0, "Sample", np.arange(1, len(X_val) + 1))

    # Reorder columns to place "Predicted Class" as the second column
    columns_order = ["Sample", "Predicted Class"] + column_names
    validation_results = validation_results[columns_order]

    # Save results to a CSV file
    output_path = folder.create_folder_get_output_path(
        "PLS_DA", validation_csv_file, suffix="validation_with_probabilities", ext="csv", validation=True
    )
    validation_results.to_csv(output_path, index=False)

    return validation_results
