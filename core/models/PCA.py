import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from core import folder
from core import preprocess
import matplotlib.pyplot as plt

# Load SAF-CAF data
# feature_file_paths = ["outputs/CBFV/mat2vec.csv","outputs/CAF/features_binary.csv"]
feature_file_paths = ["outputs/CAF/features_binary.csv"]

for feature_file_path in feature_file_paths:

    # Load the dataset
    _, X, _ = preprocess.prepare_standarlize_X_block_(feature_file_path)
    df_SAF = pd.read_csv("data/features.csv")
    y = df_SAF["Structure"]

    # Initialize and fit the LabelEncoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Applying PCA
    pca = PCA(n_components=2)  # Adjust the number of components as needed
    X_pca = pca.fit_transform(X)  # Fit and transform X

    # Define the file path for saving the plot
    plot_path = folder.create_folder_get_output_path(
        "PCA_plot",
        feature_file_path,
        "n=2",
        "png",
    )

    # Optionally, check the explained variance ratio
    explained_variance_X = pca.explained_variance_ratio_
    unique = np.unique(y_encoded)
    colors = [plt.cm.jet(float(i) / max(unique)) for i in unique]

    # Plotting
    with plt.style.context("ggplot"):
        for i, label in enumerate(unique):
            mask = y_encoded == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                color=colors[i],
                s=100,
                edgecolors="k",
                label=encoder.inverse_transform([label])[0],
            )
        plt.xlabel(f"LV 1 ({(explained_variance_X[0] * 100):.2f} %)")
        plt.ylabel(f"LV 2 ({(explained_variance_X[1] * 100):.2f} %)")
        plt.legend(loc="lower center")
        plt.savefig(plot_path, dpi=300)
        # plt.show()
        plt.close()
