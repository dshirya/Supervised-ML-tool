# SAF CAF classification performance

This repository contains model performance on crystal structure classification for binary compounds, derived from 1,400 .cif files using features generated with SAF and CAF.

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

Check the `outputs` folder for ML reports, plots, etc.

For Figures 8 and 9, run:
`python figure_8_case_study.py` and
`python figure_9_case_study.py`

## Result

Our SAF+CAF features does a great job with classifying crystal structuree for intermetallic binary compouds. 

This a PLS-DA Component N=2 result for crystal structure that you can find under `outputs/SAF_CAF/PLS_DA_plot` 

![](img/SAF_CAF_binary_features_n=2.png)

- Compositional features were created using [CAF](https://github.com/bobleesj/composition-analyzer-featurizer). Ex) `outputs/CAF/features_binary.csv`
- Structural features were created using [SAF](https://github.com/bobleesj/structure-analyzer-featurizer) Ex) `outputs/SAF/binary_features.csv`

## To customize for your data

1. Place a `features.csv` file in the `data` folder. It should have a "Structure" column, from which we'll extract all "y" values.
2. Place a CSV file with features in a subdirectory within `outputs`. Example: `outputs/SAF_CAF/binary_features.csv`

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
