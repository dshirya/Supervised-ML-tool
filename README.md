### This is a copy of the repo made by @bobleesj
### Original repo: https://github.com/bobleesj/SAF-CAF-performance


## Machine Learning Models
### SVM
SVM is used for classification with an RBF kernel and probability estimates enabled.
* Hyperparameters:
  * kernel="rbf": Uses a Radial Basis Function (RBF) kernel for non-linear classification.
  * probability=True: Enables probability estimates using Platt scaling.
  * random_state=41: Ensures reproducibility.
* Class Label Encoding:
  * Uses predefined class mapping:
```
{"Cu3Au": 1, "Cr3Si": 2, "PuNi3": 3, "Fe3C": 4, "Mg3Cd": 5, "TiAl3": 6}
```
*	Model Evaluation and Outputs:
  *	Uses 10-Fold Stratified Cross-Validation to ensure balanced splits across all classes.
  *	Generates a classification report with precision, recall, and F1-score.
  *	Predictions for the validation dataset include probability estimates for each class.
* Output Files:
  * SVM_validation_with_probabilities.csv: Stores validation predictions and probability scores.
### PLS-DA
PLS-DA is used for supervised classification by projecting predictor variables (X) and response variables (y) into a lower-dimensional space. The number of components (n_components) is dynamically selected using cross-validation.
* Hyperparameters:
  * n_components: Automatically determined between 2 and 10 via Stratified 10-Fold Cross-Validation.
  * scale=False: Disables internal scaling to retain the original distribution of input features.
* Class Label Encoding:
  * Class labels are assigned numerical values based on silhouette scores:
```
{'Cu3Au': 2, 'Cr3Si': 4, 'PuNi3': 6, 'Fe3C': 3, 'Mg3Cd': 1, 'TiAl3': 5}
``` 
* Output Files:
  * PLS_DA_n_analysis.csv: Accuracy for different n_components.
  *	PLS_DA_feature_importance.csv: Feature importance scores.
  *	PLS_DA_validation_with_probabilities: Probability scores for each class. Validation data is transformed and predictions are saved with one-vs-rest probabilities.

### XGBooost
XGBoost is used for classification with optimized hyperparameters.
* Hyperparameters:
  * eval_metric="mlogloss": Uses Multiclass Logarithmic Loss for evaluation.
  * random_state=19: Ensures consistent training results.
* Class Label Encoding:
  * XGBoost requires labels to start from 0, so class mapping is adjusted:
```
y_encoded_zero_based = np.array(y_encoded) - 1
```
```
{"Cu3Au": 1, "Cr3Si": 2, "PuNi3": 3, "Fe3C": 4, "Mg3Cd": 5, "TiAl3": 6}
```
* Model Evaluation and Outputs:
  * Performs 10-Fold Stratified Cross-Validation for performance assessment.
  * Extracts feature importance scores using the gain metric.
  * Saves top 10 most important features based on their contribution to predictions.
  * Validation results include predicted classes and probability scores.
* Output Files:
  * XGBoost_gain_score.png: Feature importance plot.
  * XGBoost_validation_with_probabilities.csv: Validation results with probability scores.
 
### Directory structure
```
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
│   ├── SVM/
│   │   ├── training/
│   │   ├── validation/
│   ├── XGBoost/
│   │   ├── training/
│   │   ├── validation/
```
* Training outputs are stored in outputs/{model_name}/training/.
* Validation results are saved separately in outputs/{model_name}/validation/.

## How to reproduce

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

For help with generating structural data using SAF, contact Bob at [sl5400@columbia.edu](mailto:sl5400@columbia.edu).
