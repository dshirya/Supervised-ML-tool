import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from core import folder

# Define the class-to-integer mapping
class_mapping = {'Cu3Au': 2, 'Cr3Si': 4, 'PuNi3': 6, 'Fe3C': 3, 'Mg3Cd': 1, 'TiAl3': 5}  #based on Silhouettes scores

def encode_classes(y, class_mapping):
    """Encodes the class labels using the provided mapping."""
    return np.array([class_mapping[label] for label in y])

def format_formula_to_latex(formula):
    """
    Convert a chemical formula string into LaTeX format with subscripts. d
    E.g., "H2O" -> "H$_2$O"
    """
    formatted = ""
    for char in formula:
        if char.isdigit():  # Convert digits to subscript
            formatted += f"$_{char}$"
        else:
            formatted += char
    return formatted

def plot_two_component(X, y, output_file_path):
    # Encode class labels using the predefined mapping
    y_encoded = encode_classes(y, class_mapping)

    # Load the dataset into PLS
    pls = PLSRegression(n_components=2)
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
        output_file_path,
        "n=2",
        ext = "png",
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
                label=format_formula_to_latex(list(class_mapping.keys())[list(class_mapping.values()).index(label)]),
            )

        plt.xlabel(f"LV 1 ({(explained_variance_X[0] * 100):.2f} %)")
        plt.ylabel(f"LV 2 ({(explained_variance_X[1] * 100):.2f} %)")
        plt.legend(loc="lower left", fontsize=8, title_fontsize=9)
        # plt.title(f"PLS Cross-Decomposition")
        plt.savefig(plot_path, dpi=600)  # Save the plot as a PNG file
        plt.close()
        # plt.show()

def plot_two_component_with_validation(X, y, X_val, validation_csv_file):
    # Encode class labels using the predefined mapping
    y_encoded = encode_classes(y, class_mapping)

    # Train the PLS model with 2 components
    pls = PLSRegression(n_components=2, scale=False)
    X_pls = pls.fit_transform(X, y_encoded)[0]  # Extract X_scores
    X_val_pls = pls.transform(X_val)  # Transform validation data into the same latent space

    # Calculate the variance explained by each component for X
    total_variance_X = np.var(X, axis=0).sum()
    explained_variance_X = [
        np.var(X_pls[:, i]) / total_variance_X for i in range(pls.n_components)
    ]

    plot_path = folder.create_folder_get_output_path(
        "PLS_DA_plot", validation_csv_file, suffix="validation", ext="png", validation=True
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
        # Plot training data
        for i, label in enumerate(unique):
            xi = [X_pls[j, 0] for j in range(len(X_pls[:, 0])) if y_encoded[j] == label]
            yi = [X_pls[j, 1] for j in range(len(X_pls[:, 1])) if y_encoded[j] == label]
            plt.scatter(
                xi,
                yi,
                color=colors[i],
                s=50,
                edgecolors="k",
                label=format_formula_to_latex(list(class_mapping.keys())[list(class_mapping.values()).index(label)]),
            )

        # Plot validation data
        plt.scatter(
            X_val_pls[:, 0],
            X_val_pls[:, 1],
            color="white",
            s=70,
            edgecolors="black",
            marker="*",
            linewidths=1,
            label="Test Data",
        )

        plt.xlabel(f"LV 1 ({(explained_variance_X[0] * 100):.2f} %)")
        plt.ylabel(f"LV 2 ({(explained_variance_X[1] * 100):.2f} %)")
        plt.legend(loc="lower left", fontsize=8, title_fontsize=9)
        plt.title(f"PLS-DA Scatterplot: Training and Test Data")  # noqa: F541
        plt.savefig(plot_path, dpi=600)  # Save the plot as a PNG file
        plt.close()