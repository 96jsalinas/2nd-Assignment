# Bank Marketing Dataset - Classification Analysis

**Advanced Programming 2nd Assignment**  
**Student ID:** 100562234  
**Python Version:** 3.13.5 (Conda)  
**IDE:** Positron

## Project Overview

This project analyzes the Bank Marketing dataset to predict whether a client will subscribe to a term deposit (`deposit` target variable). The analysis includes exploratory data analysis, model comparison (Decision Trees, KNN), hyperparameter optimization, and an open-choice task using CatBoost.

### Key Results

| Model | Test Accuracy |
|-------|---------------|
| CatBoost (Task 5) | 87.23% |
| Decision Tree (Tuned) | 82.95% |
| KNN (Tuned) | 82.18% |
| KNN (Default) | 81.18% |
| Decision Tree (Default) | 79.86% |
| Dummy Baseline | 52.55% |

**Best Model:** Decision Tree (Tuned) was selected as the final model for Task 4, with CatBoost demonstrating superior performance in the open-choice task.

## Project Structure

```
├── data/
│   ├── bank_34.pkl              # Main dataset (11,000 instances, 16 features)
│   └── bank_competition.pkl     # Competition dataset for predictions
├── src/
│   ├── main.py                  # Main entry point - runs all tasks
│   ├── task1_eda.py             # Exploratory Data Analysis
│   ├── task2_evaluation.py      # Train/test split and preprocessing pipelines
│   ├── task3_trees_knn.py       # Decision Trees and KNN comparison
│   ├── task4_results.py         # Final model training and competition predictions
│   ├── task5_open_choice.py     # CatBoost implementation
│   └── task6_feature_selection.py  # Feature selection for KNN (optional)
├── outputs/
│   ├── final_model.joblib       # Saved final model
│   └── competition_predictions.csv  # Competition predictions
├── report/
│   ├── main.tex                 # Main LaTeX report
│   ├── task1_eda.tex            # EDA report section
│   ├── task2_evaluation.tex     # Evaluation setup report section
│   ├── task3_trees_knn.tex      # Model comparison report section
│   ├── task4_results.tex        # Results report section
│   ├── task5_open_choice.tex    # CatBoost report section
│   ├── task6_feature_selection.tex  # Feature selection report section
│   └── figures/                 # Generated figures for report
├── requirements.MD              # Assignment requirements
└── README.md                    # This file
```

## Installation & Setup

### Prerequisites

- Python 3.13+ (Anaconda/Conda recommended)
- Required packages: scikit-learn, pandas, numpy, catboost, matplotlib, joblib

### Install Dependencies

```bash
pip install scikit-learn pandas numpy catboost matplotlib joblib
```

Or with Conda:

```bash
conda install scikit-learn pandas numpy matplotlib joblib
pip install catboost
```

## Usage

### Run All Tasks

Execute all tasks sequentially from the project root:

```bash
cd src
python main.py
```

This will:
1. Perform EDA on the dataset
2. Set up train/test split (80/20, stratified)
3. Train and compare Decision Trees and KNN models
4. Select best model and generate competition predictions
5. Train CatBoost as open-choice task
6. Run feature selection analysis for KNN

### Run Individual Tasks

Each task can be run independently:

```bash
python task1_eda.py          # EDA only
python task3_trees_knn.py    # Model comparison only
python task5_open_choice.py  # CatBoost only
```

### Load the Final Model

```python
import joblib

# Load the trained model
model = joblib.load('outputs/final_model.joblib')

# Make predictions
predictions = model.predict(X_new)
```

## Tasks Summary

### Task 1: Exploratory Data Analysis
- 11,000 instances, 16 features
- 7 numerical, 10 categorical variables
- No missing values, no constant/ID columns
- Binary classification (deposit: yes/no)
- Balanced classes (52.55% no, 47.45% yes)
- Special handling for `pdays` (-1 indicates no previous contact)

### Task 2: Evaluation Setup
- 80/20 train/test split with stratification
- 3-fold Stratified Cross-Validation for inner evaluation
- Preprocessing pipeline: One-Hot Encoding + Standard Scaling + pdays transformation

### Task 3: Decision Trees & KNN Comparison
- Compared StandardScaler vs MinMaxScaler for KNN → StandardScaler selected
- GridSearchCV for hyperparameter tuning
- Decision Tree (Tuned): 82.95% accuracy
- KNN (Tuned): 82.18% accuracy

### Task 4: Final Model
- Best model: Decision Tree with `max_depth=10`, `criterion='gini'`
- Model trained on full dataset and saved
- Competition predictions generated

### Task 5: CatBoost (Open Choice)
- Native categorical feature handling
- 87.23% accuracy (+5.14% over baseline)
- Early stopping with 50-round patience

### Task 6: Feature Selection (Optional)
- SelectKBest with ANOVA F-value
- All 53 features retained as optimal
- No dimensionality reduction beneficial for KNN

## Random Seed

All experiments use student ID `100562234` as the random seed for reproducibility.

## License

This project is for educational purposes as part of the Advanced Programming course at UC3M.
