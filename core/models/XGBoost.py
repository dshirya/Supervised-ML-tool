import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from core import folder


def run_XGBoost(X_df, y):
    # Initialize the Label Encoder and encode the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)  # Do not shift labels here

    # Initialize Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)

    # Initialize the XGBoost Classifier
    model = XGBClassifier(eval_metric="mlogloss")

    # Cross-validate and get predictions for each fold
    y_pred = cross_val_predict(model, X_df, y_encoded, cv=skf)

    # Decode predicted labels back to original (shifted for reporting purposes)
    y_pred_decoded = encoder.inverse_transform(y_pred)

    # Evaluate the model
    class_report = classification_report(
        y, y_pred_decoded, digits=3, output_dict=True, labels=encoder.classes_
    )
    return class_report

def plot_XGBoost_feature_importance(X_df, y_encoded, csv_file_path):
    model = XGBClassifier(eval_metric="mlogloss")
    # Fit the model to the entire dataset to retrieve feature importances
    model.fit(X_df, y_encoded)

    # Assuming gain_importances is already retrieved from the model
    gain_importances = model.get_booster().get_score(importance_type="gain")

    # Convert to series and sort
    gain_features = pd.Series(gain_importances)
    gain_features = gain_features.sort_values(ascending=True)

    # Select the top 10 features
    top_gain_features = gain_features.tail(
        10
    )  # Since it's sorted ascending, tail will give the largest

    # Create a horizontal bar plot with adjusted dimensions and spacing
    plt.figure(figsize=(8, 8))  # Adjust figure size for the number of features
    ax = plt.subplot(111)  # Add a subplot to manipulate the space for the axis

    # Plotting
    top_gain_features.plot(kind="barh", color="skyblue", ax=ax)

    output_path = folder.create_folder_get_output_path(
        "XGBoost",
        csv_file_path,
        suffix="gain_score",
        ext="png",
    )
    # Adjust left margin to make more space for long labels
    plt.subplots_adjust(
        left=0.5
    )  # Adjust this value based on your actual label lengths

    plt.xlabel("Gain Score")  # X-axis label for scores
    plt.title("Top 10 Feature Importances by Gain")
    # Save high qualitty image with tight layout
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show()

    plt.close()

def validate_XGBoost(X_train, y_train, X_val, csv_file_path, validation_csv_file):
    """
    Validates the XGBoost model on a validation dataset and includes predicted probabilities for each class.

    Parameters:
        X_train: Training feature matrix.
        y_train: Training target labels.
        X_val: Validation feature matrix.
        csv_file_path: Path of the training data (for output organization).
        validation_csv_file: Path of the validation data.

    Returns:
        y_pred_validation: Predicted classes for the validation set.
        probabilities: Predicted probabilities for each class.
    """
    # Initialize the Label Encoder and encode the training labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)  # Keep labels starting from 0 for XGBoost

    # Initialize the XGBoost Classifier
    model = XGBClassifier(eval_metric="mlogloss", random_state=19)

    # Fit the model to the training data
    model.fit(X_train, y_train_encoded)

    # Predict probabilities for the validation set
    probabilities = model.predict_proba(X_val)

    # Predicted class labels (0-based)
    y_pred_encoded = probabilities.argmax(axis=1)

    # Shift the predicted class labels to start from 1
    y_pred_validation = y_pred_encoded + 1

    # Dynamically determine the number of classes from probabilities
    n_classes = probabilities.shape[1]

    # Prepare the DataFrame for output
    validation_results = pd.DataFrame(
        {
            "Validation Sample": range(1, len(X_val) + 1),
            "Predicted Class": y_pred_validation,  # Predicted class as a number starting from 1
        }
    )

    # Add probability columns for each class (shift class indices to start from 1)
    for i in range(n_classes):
        validation_results[f"Class_{i + 1}_Probability"] = probabilities[:, i]

    # Save predictions and probabilities to a CSV file
    output_path = folder.create_folder_get_output_path(
        "XGBoost",
        validation_csv_file,
        suffix="validation_predictions_with_probabilities",
        ext="csv",
    )
    validation_results.to_csv(output_path, index=False)

    return y_pred_validation, probabilities