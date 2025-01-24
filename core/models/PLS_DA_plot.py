import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

from core import folder


def plot_two_component(X, y, feature_file_path):
    # Convert string labels to integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Load the dataset into PLS
    pls = PLSRegression(n_components=2, scale=False)
    # fit_transform returns a tuple (X_scores, Y_scores)
    X_pls = pls.fit_transform(X, y_encoded)[0]

    # Calculate the variance explained by each component for X
    total_variance_X = np.var(X, axis=0).sum()

    # calculates the variance of the scores for the
    explained_variance_X = [
        np.var(X_pls[:, i]) / total_variance_X for i in range(pls.n_components)
    ]

    # Define the file path for saving the plot
    plot_path = folder.create_folder_get_output_path(
        "PLS_DA_plot",
        feature_file_path,
        "n=2",
        "png",
    )

    # Scatter plot
    unique = np.unique(y_encoded)
    colors = [
        "#c3121e",  # Sangre
        "#0348a1",  # Neptune
        "#ffb01c",  # Pumpkin
        "#027608",  # Clover
        "#1dace6",  # Cerulean
        "#9c5300",  # Cocoa
        "#9966cc",  # Amethyst
        "#ff4500",  # Orange Red
    ]

    with plt.style.context("ggplot"):
        for i, label in enumerate(unique):
            xi = [X_pls[j, 0] for j in range(len(X_pls[:, 0])) if y_encoded[j] == label]
            yi = [X_pls[j, 1] for j in range(len(X_pls[:, 1])) if y_encoded[j] == label]
            plt.scatter(
                xi,
                yi,
                color=colors[i],
                s=50,
                edgecolors="k",
                label=encoder.inverse_transform([label])[0],
            )

        plt.xlabel(f"LV 1 ({(explained_variance_X[0] * 100):.2f} %)")
        plt.ylabel(f"LV 2 ({(explained_variance_X[1] * 100):.2f} %)")
        plt.legend(loc="lower left", fontsize=8)
        # plt.title(f"PLS Cross-Decomposition")
        plt.savefig(plot_path, dpi=500)  # Save the plot as a PNG file
        plt.close()
        # plt.show()
