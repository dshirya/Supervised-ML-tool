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
    y_encoded = encoder.fit_transform(y)

    # Initialize Stratified K-Fold cross-validator
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)

    # Initialize the XGBoost Classifier
    model = XGBClassifier(eval_metric="mlogloss")

    # Cross-validate and get predictions for each fold
    y_pred = cross_val_predict(model, X_df, y_encoded, cv=skf)

    # Decode predicted labels back to original
    y_pred_decoded = encoder.inverse_transform(y_pred)

    # Evaluate the model
    class_report = classification_report(y, y_pred_decoded, digits=3, output_dict=True)
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
