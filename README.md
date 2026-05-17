# AI_Bootcamp

This repository contains a set of Python machine learning practice scripts built with scikit-learn, pandas, and related data science libraries.

## Contents

- `day2_ex.py` - Iris classification with k-NN, including a comparison between raw features and MinMax scaling.
- `day3_ex.py` - Titanic survival prediction using one-hot encoding and logistic regression.
- `day4_ex.py` - Diabetes feature exploration using correlation, mutual information, and random forest feature importance.
- `day5_ex.py` - Bike sharing regression experiment with polynomial features and linear regression.
- `day6_ex.py` - Binary Iris classification with logistic regression and a confusion matrix report.
- `day6_ex2.py` - California housing regression using linear regression and common error metrics.
- `day7_ex.py` - Titanic preprocessing with scaling and encoding, followed by cross-validation and random forest grid search.
- `day8_ex.py` - Iris ensemble classification using voting with logistic regression, decision tree, and k-NN.
- `bike_sharing_daily.csv` - Dataset used by the bike sharing exercise.

## Setup

1. Create and activate a virtual environment if you want to keep dependencies isolated.
2. Install the Python packages:

```bash
pip install -r requirements.txt
```

## Run

Run any script directly with Python:

```bash
python day8_ex.py
```

Replace `day8_ex.py` with the file you want to execute.

## Notes

- Some scripts download datasets from the internet when they run.
- `day4_ex.py` and `day5_ex.py` include plots or commented visualization code.
- `day5_ex.py` is currently fully commented out.
- Several scripts print model metrics directly to the console.