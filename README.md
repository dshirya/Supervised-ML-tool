## This is a copy of the repo made by @bobleesj
## Original repo: https://github.com/bobleesj/SAF-CAF-performance


## How to reproduce

```bash
# Download the repository
git clone https://www.github.com/bobleesj/CAF_SAF_perfomance

# Enter the folder
cd CAF_SAF_perfomance
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
imac@imacs-iMac digitial-discovery % python main.py

Processing outputs/CAF/features_binary.csv with 133 features (1/7).
(1/4) Running SVM model...
(2/4) Running PLS_DA n=2...
(3/4) Running PLS_DA model with the best n...
(4/4) Running XGBoost model...
===========Elapsed time: 8.30 seconds===========

...

Processing outputs/CBFV/oliynyk.csv with 308 features (7/7).
(1/4) Running SVM model...
(2/4) Running PLS_DA n=2...
(3/4) Running PLS_DA model with the best n...
(4/4) Running XGBoost model...
===========Elapsed time: 12.88 seconds===========
imac@imacs-iMac digitial-discovery % 
```

Check the `features_results` folder for ML reports, plots, etc.


## To customize for your data

1. Place a file with class information in the `data` folder. It should have a "Structure" column, from which we'll extract all "y" values.
2. Place a CSV file with features in a subdirectory within `features_results`. Example: `features_results/SAF_CAF/binary_features.csv`

## To format the code

To automatically format Python code and organize imports:

```bash
black -l 79 . && isort .
```

## To generate features with CBFV

Run the following command:

```bash
python featurizer.py
```

## Questions?

For help with generating structural data using SAF, contact Bob at [sl5400@columbia.edu](mailto:sl5400@columbia.edu).
