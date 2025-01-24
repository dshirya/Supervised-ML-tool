import os

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core import folder, preprocess


def get_report(X, y):
    # Define the model
    model = SVC(kernel="rbf")

    # StratifiedKFold to ensure balanced folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=41)

    # Use cross_val_predict to get predictions for each fold
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Evaluating the model, get results as a dictionary
    report_dict = classification_report(y, y_pred, digits=3, output_dict=True)
    return report_dict
