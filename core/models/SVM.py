import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from core import folder

# Define the class-to-integer mapping
class_mapping = {
    "Cu3Au": 1,
    "Cr3Si": 2,
    "PuNi3": 3,
    "Fe3C": 4,
    "Mg3Cd": 5,
    "TiAl3": 6,
}

def encode_classes(y, class_mapping):
    """Encodes the class labels using the provided mapping."""
    return [class_mapping[label] for label in y]

def decode_classes(y_encoded, class_mapping):
    """Decodes the integer class labels back to their string representation."""
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    return [reverse_mapping[label] for label in y_encoded]

def get_report(X, y):
    # Encode class labels using the predefined mapping
    y_encoded = encode_classes(y, class_mapping)

    # Define the model
    model = SVC(kernel="rbf")

    # StratifiedKFold to ensure balanced folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=41)

    # Use cross_val_predict to get predictions for each fold
    y_pred = cross_val_predict(model, X, y_encoded, cv=skf)

    # Evaluating the model, get results as a dictionary
    report_dict = classification_report(y_encoded, y_pred, digits=3, output_dict=True)
    return report_dict

def validate_svc_with_probabilities(X_train, y_train, X_val, csv_file_path, validation_csv_file):
    """
    Validates the SVC model on a validation dataset and includes probabilities for each class.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        X_val: Validation feature matrix.
        csv_file_path: Path of the training data (for output organization).
        validation_csv_file: Path of the validation data.

    Returns:
        probabilities: Probabilities for each class on the validation set.
        y_pred: Predicted classes on the validation set.
    """
    # Encode class labels using the predefined mapping
    y_train_encoded = encode_classes(y_train, class_mapping)

    # Define the SVC model with probability=True
    model = SVC(kernel="rbf", probability=True, random_state=41)

    # Fit the model on the training data
    model.fit(X_train, y_train_encoded)

    # Predict probabilities for the validation set
    probabilities = model.predict_proba(X_val)

    # Predicted class labels (0-based)
    y_pred = probabilities.argmax(axis=1) + 1  # Classes start from 1

    # Dynamically determine the number of classes from probabilities
    n_classes = probabilities.shape[1]

    # Prepare the DataFrame for output
    validation_results = pd.DataFrame(
        {
            "Validation Sample": range(1, len(X_val) + 1),  # Validation sample numbers start from 1
            "Predicted Class": y_pred,  # Predicted class labels starting from 1
        }
    )

    # Add probability columns for each class (shift column names to start from 1)
    for i in range(n_classes):
        validation_results[f"Class_{i + 1}"] = probabilities[:, i]

    # Save predictions and probabilities to a CSV file
    output_path = folder.create_folder_get_output_path(
        "SVM",
        validation_csv_file,
        suffix="validation_with_probabilities",
        ext="csv",
        validation=True
    )
    validation_results.to_csv(output_path, index=False)

    #print(f"Validation predictions with probabilities saved to {output_path}")
    return probabilities, y_pred
