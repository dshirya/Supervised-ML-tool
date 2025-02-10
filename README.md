# Supervised-ML-tool 

This project utilizes multiple machine learning models for classification tasks related to material structures. The models include **PLS-DA** (Partial Least Squares Discriminant Analysis),  **SVM** (Support Vector Machine), and **XGBoost**. Each model is trained on labeled datasets and validated using a separate validation datase

This project is a modification of the [CAF-SAF-performance](https://github.com/bobleesj/SAF-CAF-performance) repository, which was developed to evaluate the classification performance of crystal structures using **SAF** (Structure Analyzer/Featurizer) and **CAF** (Composition Structure Analyzer/Featurizer).

The methods implemented in this project are based on the work presented in the following publication:

📄 [Composition and structure analyzer/featurizer for explainable machine-learning models to predict solid state structures](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00332b)

If you use this repository or its modifications in your research, please consider citing the original work.

# Machine Learning Models
## SVM
SVM is used for classification with an RBF kernel and probability estimates enabled.
* **Hyperparameters**:
  * `kernel="rbf"`: Uses a Radial Basis Function (RBF) kernel for non-linear classification.
  * `probability=True`: Enables probability estimates using Platt scaling.
  * `random_state=41`: Ensures reproducibility.
* **Class Label Encoding**:
  * Uses predefined class mapping:
```bash
{"Cu3Au": 1, "Cr3Si": 2, "PuNi3": 3, "Fe3C": 4, "Mg3Cd": 5, "TiAl3": 6}
```
*	**Model Evaluation and Outputs**:
  	* Uses 10-Fold Stratified Cross-Validation to ensure balanced splits across all classes.
   	* Generates a classification report with precision, recall, and F1-score.
    * Predictions for the validation dataset include probability estimates for each class.
* **Output Files**:
  * SVM_validation_with_probabilities.csv: Stores validation predictions and probability scores.
  * SVM_report.csv: Classification report with precision, recall, and F1-score.
## PLS-DA
PLS-DA is used for supervised classification by projecting predictor variables (X) and response variables (y) into a lower-dimensional space. The number of components (*n_components*) is dynamically selected using cross-validation.
* **Hyperparameters**:
  * `n_components`: Automatically determined between 2 and 10 via Stratified 10-Fold Cross-Validation.
  * `scale=False`: Disables internal scaling to retain the original distribution of input features.
* **Class Label Encoding**:
  
The selection of an appropriate labeling scheme for the Partial Least Squares Discriminant Analysis (PLS-DA) model is a critical step in ensuring accurate classification and optimal separation between classes in the latent space. Since PLS-DA treats class labels as numerical values, the choice of numerical encoding directly influences model performance and class distribution in the reduced-dimensional space. To determine the most effective labeling, multiple approaches were evaluated based on classification metrics and clustering quality measures.

**Methods for Labeling Selection**

Several strategies were applied to assess the impact of different label assignments:

1. **F1-Score Optimization**
   - Multiple label permutations were tested to identify the assignment yielding the highest macro F1-score.
   - The F1-score accounts for both precision and recall, making it particularly relevant for imbalanced datasets.
   - However, in certain cases, the highest F1-score corresponded to labelings that produced nearly linear scatterplots, suggesting poor class separation in the latent space.
```bash
Best Mapping: {'Cu3Au': 3, 'Cr3Si': 5, 'PuNi3': 6, 'Fe3C': 1, 'Mg3Cd': 4, 'TiAl3': 2}, Macro F1-Score: 0.991
```
2. **Accuracy-Based Labeling**
   - Labelings were evaluated based on their overall classification accuracy.
   - While accuracy provides a straightforward measure of model performance, it does not necessarily reflect the quality of class separation in the PLS-DA projection.
   - The result of that type of labeling was the same as for F1-score, with nearly linear scatterplots.
```bash
Best Mapping: {'Cu3Au': 3, 'Cr3Si': 5, 'PuNi3': 6, 'Fe3C': 1, 'Mg3Cd': 4, 'TiAl3': 2} with Accuracy: 0.993
```
3. **Fisher’s Discriminant Ratio (FDR) Optimization**
```python
def fisher_discriminant_ratio(X_pls, y_encoded):
    """Calculate Fisher's Discriminant Ratio (FDR)."""
    classes = np.unique(y_encoded)
    overall_mean = np.mean(X_pls, axis=0)
    between_class_variance = 0
    within_class_variance = 0

    for cls in classes:
        class_data = X_pls[y_encoded == cls]
        class_mean = np.mean(class_data, axis=0)
        between_class_variance += len(class_data) * np.sum((class_mean - overall_mean)**2)
        within_class_variance += np.sum((class_data - class_mean)**2)

    return between_class_variance / within_class_variance
```
   - Labelings were evaluated using Fisher’s Discriminant Ratio, which measures the separation between class distributions relative to their within-class variance.
   - Higher FDR values indicate better class separation, making it a useful metric for optimizing label assignments.
   - However, in some cases, maximizing FDR did not consistently lead to visually well-separated clusters.
```bash
Best Mapping: {'Cu3Au': 2, 'Cr3Si': 3, 'PuNi3': 6, 'Fe3C': 4, 'Mg3Cd': 1, 'TiAl3': 5} with FDR Value: 17.044
```
4. **Silhouette Score Maximization**
```python
# Compute silhouette score
silhouette = silhouette_score(X_pls, y_encoded)
print(f"Silhouette Score: {silhouette}")
```
	  -	Silhouette analysis was applied to measure the cohesion and separation of clusters in the PLS-DA latent space.
   -	The silhouette score quantifies how well a sample is clustered within its assigned class while distinguishing it from other classes.
   -	The labeling with the highest silhouette score exhibited the most well-defined clusters, indicating strong inter-class separation and intra-class cohesion.
```bash
Best Mapping: {'Cu3Au': 2, 'Cr3Si': 4, 'PuNi3': 6, 'Fe3C': 3, 'Mg3Cd': 1, 'TiAl3': 5} with Silhouette Value: 0.640
```
5. **Pairwise Distance Analysis**
```python
def pairwise_class_distances(X_pls, y_encoded):
    classes = np.unique(y_encoded)
    centroids = {cls: np.mean(X_pls[y_encoded == cls], axis=0) for cls in classes}
    
    distances = []
    for i, cls1 in enumerate(classes):
        for cls2 in classes[i+1:]:
            dist = euclidean(centroids[cls1], centroids[cls2])
            distances.append((cls1, cls2, dist))
    
    return distances
```
   - Pairwise distances between class centroids in the PLS-DA space were computed to assess the degree of separation among different class label assignments.
   - This approach identified labelings that maximized inter-class distances while maintaining intra-class compactness.
   - The results of this approach have the largest LV values.
```bash
Best Mapping: {'Cu3Au': 1, 'Cr3Si': 2, 'PuNi3': 6, 'Fe3C': 5, 'Mg3Cd': 3, 'TiAl3': 4} with Pairwise Value: 1.995
```

 <div align="center">
  
   ### Comparison of Labeling Strategies for PLS-DA: Impact on Class Separation
<img src="https://github.com/user-attachments/assets/e4b9dff4-dd12-4218-8e2d-d17720f99804" alt="labeling" width="800">
</div>

**Selection of the Optimal Labeling**

Among the evaluated methods, the labeling based on **Silhouette score** maximization was selected as the most effective due to its ability to:
- Produce distinct and well-separated clusters in the PLS-DA scatterplots.
- Provide a quantitative criterion for selecting an optimal numerical encoding.
- Balance classification performance with improved interpretability in the latent space.

```bash
#Labeling used in the analysis
{'Cu3Au': 2, 'Cr3Si': 4, 'PuNi3': 6, 'Fe3C': 3, 'Mg3Cd': 1, 'TiAl3': 5}
``` 
* **Output Files**:
  * PLS_DA_n_analysis.csv: Accuracy for different n_components.
  *	PLS_DA_feature_importance.csv: Feature importance scores.
  *	PLS_DA_validation_with_probabilities.csv: Probability scores for each class. Validation data is transformed and predictions are saved with one-vs-rest probabilities.
  *	PLS_DA_report.csv: Classification report with precision, recall, and F1-score.
  *	PLS_DA_plot_n=2.png: Classification scatterplot
  *	PLS_DA_plot_validation.png: Classification scatterplot with validation data

## XGBooost
XGBoost is used for classification with optimized hyperparameters.
* **Hyperparameters**:
  * eval_metric="mlogloss": Uses Multiclass Logarithmic Loss for evaluation.
  * random_state=19: Ensures consistent training results.
* **Class Label Encoding**:
  * XGBoost requires labels to start from 0, so class mapping is adjusted:
```bash
y_encoded_zero_based = np.array(y_encoded) - 1
```
```bash
{"Cu3Au": 1, "Cr3Si": 2, "PuNi3": 3, "Fe3C": 4, "Mg3Cd": 5, "TiAl3": 6}
```
* **Model Evaluation and Outputs**:
  * Performs 10-Fold Stratified Cross-Validation for performance assessment.
  * Extracts feature importance scores using the gain metric.
  * Saves top 10 most important features based on their contribution to predictions.
  * Validation results include predicted classes and probability scores.
* **Output Files**:
  * XGBoost_gain_score.png: Feature importance plot.
  * XGBoost_validation_with_probabilities.csv: Validation results with probability scores.
  * XGBoost_report.csv: Classification report with precision, recall, and F1-score.
  
## Directory structure
```bash
├── main.py
├── core/
│   ├── folder.py       # Handles output file management
│   ├── preprocess.py   # Data preprocessing functions
│   ├── report.py       # Model evaluation and reporting
│   ├── models/
│   │   ├── PLS_DA.py   # PLS-DA model
│   │   ├── SVM.py      # SVM model
│   │   ├── XGBoost.py  # XGBoost model
│   │   ├── PLS_DA_plot.py # Visualization for PLS-DA
│   │   ├── XGBoost_plot.py # Feature importance for XGBoost
│   ├── data/
│   │   ├── class.csv        # Class/cluster dataset
│   │   ├── validation.csv   # Validation dataset
│   ├── features/
│   │   ├── features.csv        # Features dataset
│── outputs/
│   ├── PLS_DA/
│   │   ├── training/
│   │   ├── validation/
│   ├── PLS_DA_plot/
│   │   ├── training/
│   │   ├── validation/
│   ├── SVM/
│   │   ├── training/
│   │   ├── validation/
│   ├── XGBoost/
│   │   ├── training/
│   │   ├── validation/
```
* Training outputs are stored in outputs/{model_name}/training/.
* Validation results are saved separately in outputs/{model_name}/validation/.

# How to reproduce

```bash
# Download the repository
git clone https://github.com/dshirya/Supervised-ML-tool

# Enter the folder
cd Supervised-ML-tool
```

Install packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or you may install all packages at once:

```bash
pip install matplotlib scikit-learn pandas CBFV numpy
```

## To reproduce results

Run `python main.py`

```
Processing Supervised-ML-tool/features/features.csv with 97 features (1/1).
(1/4) Running SVM model...
(2/4) Running PLS_DA n=2...
(3/4) Running PLS_DA model with the best n...
(4/4) Running XGBoost model...
===========Elapsed time: 6.38 seconds===========
```

Check the `outputs` folder for ML reports, plots, etc.


## To customize for your data

1. Place a file with class information in the `data` folder. It should have a "Structure" column, from which we'll extract all "y" values.
2. Place a CSV file with features in the `feature` folder. 


## Questions?

For help with generating structural data using SAF and CAF, contact Bob at [sl5400@columbia.edu](mailto:sl5400@columbia.edu).

For help with using that code, contact Danila at myhunter [danila.shiryaev44@myhunter.cuny.edu](mailto:danila.shiryaev44@myhunter.cuny.edu).

