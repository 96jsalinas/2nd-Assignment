"""
Task 5: Open-Choice Task - CatBoost Implementation

This module uses CatBoostClassifier to leverage the dataset's categorical features natively.
It compares the performance against the best model from Task 4.
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from task2_evaluation import load_and_prepare_data, SEED, TEST_SIZE

def prepare_data_for_catboost(data_path: str):
    """
    Load and prepare data specifically for CatBoost.
    Retains categorical features as strings/objects.
    Applies pdays transformation.
    """
    print("  Loading data for CatBoost...")
    df = pd.read_pickle(data_path)
    
    # 1. Target Encoding
    le = LabelEncoder()
    y = le.fit_transform(df['deposit'])
    X = df.drop('deposit', axis=1)
    
    # 2. Pdays Transformation (Manual implementation of Option C)
    # We want to keep dataframe structure
    pdays = X['pdays'].values
    valid_pdays = pdays[pdays != -1]
    median_pdays = np.median(valid_pdays) if len(valid_pdays) > 0 else 0
    
    X['was_contacted_before'] = (X['pdays'] != -1).astype(int)
    X['pdays_transformed'] = np.where(X['pdays'] == -1, median_pdays, X['pdays'])
    X = X.drop('pdays', axis=1)
    
    # 3. Missing Values
    # 'job' has missing values. Fill with 'unknown' for categorical handling
    X['job'] = X['job'].fillna('unknown')
    
    # 4. Identify Categorical Features
    # CatBoost expects indices or names of categorical columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Categorical features for CatBoost: {cat_features}")
    
    return X, y, cat_features, le

def run_catboost_task(data_path: str, seed: int = SEED):
    """
    Train and evaluate CatBoostClassifier.
    """
    print("\n" + "=" * 60)
    print("  TASK 5: CATBOOST (OPEN CHOICE)")
    print("=" * 60)
    
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        print("  ERROR: CatBoost is not installed. Please install it to run this task.")
        return
    
    # Prepare Data
    X, y, cat_features, le = prepare_data_for_catboost(data_path)
    
    # Split Data (Same strategy as Task 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    
    print(f"  Training Set: {X_train.shape}")
    print(f"  Test Set:     {X_test.shape}")
    
    # Initialize CatBoost
    print("\n  Training CatBoostClassifier...")
    # Using defaults which are usually very strong
    # iteration=1000, learning_rate=auto, depth=6
    clf = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='Accuracy',
        verbose=100,  # Print every 100 iterations
        random_seed=seed,
        allow_writing_files=False
    )
    
    # Fit
    start_time = time.time()
    clf.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )
    train_time = time.time() - start_time
    
    # Evaluate
    print("\n  Evaluation:")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    print("\n  Confusion Matrix:")
    print(cm)
    
    # Compare with Task 4 Baseline (approx 82.95%)
    print("-" * 30)
    print(f"  Baseline (DT Tuned): ~0.8295")
    if acc > 0.8295:
        print(f"  RESULT: IMPROVEMENT of +{(acc - 0.8295):.4f}")
    elif acc == 0.8295:
        print(f"  RESULT: TIED")
    else:
        print(f"  RESULT: NO IMPROVEMENT")
    
    return {
        'model': clf,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'train_time': train_time
    }

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_pth = os.path.join(script_dir, "..", "data", "bank_34.pkl")
    run_catboost_task(data_pth)
