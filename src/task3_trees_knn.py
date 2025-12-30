"""
Task 3: Decision Trees and KNN Comparison

This module trains, evaluates, and compares:
- Decision Trees (default hyperparameters + tuned)
- K-Nearest Neighbors (default hyperparameters + tuned)
- Dummy classifier (baseline)

KNN Requirements:
- Use pipelines with preprocessing (imputation + scaling)
- Compare 2 scaling methods to determine the best one

Hyperparameter Tuning:
- GridSearch for both Trees and KNN
- Report computational cost (execution time)

Uses: scikit-learn (pipelines, GridSearchCV, classifiers, metrics)
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import from task2 for preprocessing
from task2_evaluation import (
    create_preprocessing_pipeline, 
    create_full_pipeline,
    get_inner_cv_strategy,
    SEED
)


# =============================================================================
# Constants
# =============================================================================
N_JOBS = -1  # Use all available cores for parallel processing


# =============================================================================
# Helper Functions
# =============================================================================
def print_section(title: str, char: str = "-", width: int = 60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_results(results: Dict[str, Any], show_cm: bool = True):
    """Print model results in a formatted way."""
    print(f"  Accuracy:       {results['accuracy']:.4f}")
    print(f"  Training time:  {results['train_time']:.2f}s")
    if 'cv_mean' in results:
        print(f"  CV Mean (±std): {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
    if 'best_params' in results and results['best_params']:
        print(f"  Best params:    {results['best_params']}")
    if show_cm:
        print(f"\n  Confusion Matrix:")
        print(f"  {results['confusion_matrix']}")


# =============================================================================
# Baseline: Dummy Classifier
# =============================================================================
def train_dummy_baseline(X_train, y_train, X_test, y_test, 
                          column_info: Dict, seed: int) -> Dict[str, Any]:
    """
    Train and evaluate the Dummy classifier baseline.
    
    Uses 'most_frequent' strategy (always predicts the majority class).
    This provides a baseline that any real model should beat.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    column_info : Column type information for preprocessing
    seed : Random seed
    
    Returns
    -------
    Dict with accuracy, confusion_matrix, train_time
    """
    print_section("DUMMY CLASSIFIER (Baseline)")
    
    # Create pipeline with preprocessing + Dummy
    dummy = DummyClassifier(strategy='most_frequent', random_state=seed)
    pipeline = create_full_pipeline(column_info, dummy, scaler_type="standard")
    
    # Train with timing
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': 'Dummy (most_frequent)',
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'train_time': train_time,
        'pipeline': pipeline
    }
    
    print_results(results)
    return results


# =============================================================================
# Decision Tree Classifier
# =============================================================================
def train_decision_tree(X_train, y_train, X_test, y_test,
                         column_info: Dict, seed: int,
                         tune: bool = False, cv=None) -> Dict[str, Any]:
    """
    Train and evaluate Decision Tree classifier.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    column_info : Column type information for preprocessing
    seed : Random seed
    tune : bool
        If True, perform GridSearchCV hyperparameter tuning
    cv : Cross-validation splitter (required if tune=True)
    
    Returns
    -------
    Dict with accuracy, confusion_matrix, train_time, best_params (if tuned)
    """
    if tune:
        print_section("DECISION TREE (Tuned - GridSearchCV)")
    else:
        print_section("DECISION TREE (Default)")
    
    # Create base estimator
    dt = DecisionTreeClassifier(random_state=seed)
    
    # Create full pipeline with preprocessing
    pipeline = create_full_pipeline(column_info, dt, scaler_type="standard")
    
    if tune:
        # Hyperparameter grid for Decision Tree
        param_grid = {
            'classifier__max_depth': [3, 5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        }
        
        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=N_JOBS,
            verbose=1,
            return_train_score=True
        )
        
        # Fit with timing
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Get best model
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_mean = grid_search.best_score_
        cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        
        # Predict on test set
        y_pred = best_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': 'Decision Tree (Tuned)',
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'train_time': train_time,
            'best_params': best_params,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'pipeline': best_pipeline,
            'grid_search': grid_search
        }
    else:
        # Default parameters - just fit and evaluate
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': 'Decision Tree (Default)',
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'train_time': train_time,
            'pipeline': pipeline
        }
    
    print_results(results)
    return results


# =============================================================================
# K-Nearest Neighbors Classifier
# =============================================================================
def train_knn(X_train, y_train, X_test, y_test,
              column_info: Dict, seed: int,
              scaler_type: str = "standard",
              tune: bool = False, cv=None) -> Dict[str, Any]:
    """
    Train and evaluate KNN classifier with preprocessing pipeline.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    column_info : Column type information for preprocessing
    seed : Random seed
    scaler_type : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    tune : bool
        If True, perform GridSearchCV hyperparameter tuning
    cv : Cross-validation splitter (required if tune=True)
    
    Returns
    -------
    Dict with accuracy, confusion_matrix, train_time, best_params (if tuned)
    """
    if tune:
        print_section(f"KNN (Tuned - {scaler_type.upper()} Scaler)")
    else:
        print_section(f"KNN (Default - {scaler_type.upper()} Scaler)")
    
    # Create KNN estimator
    knn = KNeighborsClassifier()
    
    # Create full pipeline with specified scaler
    pipeline = create_full_pipeline(column_info, knn, scaler_type=scaler_type)
    
    if tune:
        # Hyperparameter grid for KNN
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 11, 15, 21],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=N_JOBS,
            verbose=1,
            return_train_score=True
        )
        
        # Fit with timing
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Get best model
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_mean = grid_search.best_score_
        cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        
        # Predict on test set
        y_pred = best_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': f'KNN (Tuned, {scaler_type})',
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'train_time': train_time,
            'best_params': best_params,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'pipeline': best_pipeline,
            'grid_search': grid_search,
            'scaler_type': scaler_type
        }
    else:
        # Default parameters - just fit and evaluate
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': f'KNN (Default, {scaler_type})',
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'train_time': train_time,
            'pipeline': pipeline,
            'scaler_type': scaler_type
        }
    
    print_results(results)
    return results


# =============================================================================
# Scaler Comparison for KNN
# =============================================================================
def compare_scalers(X_train, y_train, X_test, y_test,
                    column_info: Dict, seed: int, cv) -> Dict[str, Any]:
    """
    Compare StandardScaler vs MinMaxScaler for KNN.
    
    Uses cross-validation to determine which scaler works better.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    column_info : Column type information
    seed : Random seed
    cv : Cross-validation splitter
    
    Returns
    -------
    Dict with comparison results and best scaler
    """
    print_section("SCALER COMPARISON FOR KNN", char="=")
    
    results = {}
    
    for scaler_type in ['standard', 'minmax']:
        print(f"\n  Testing {scaler_type.upper()} Scaler...")
        
        # Create KNN with default params
        knn = KNeighborsClassifier()
        pipeline = create_full_pipeline(column_info, knn, scaler_type=scaler_type)
        
        # Cross-validation on training data
        start_time = time.time()
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, 
                                     scoring='accuracy', n_jobs=N_JOBS)
        cv_time = time.time() - start_time
        
        results[scaler_type] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_time': cv_time
        }
        
        print(f"    CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"    CV Time: {cv_time:.2f}s")
    
    # Determine best scaler
    if results['standard']['cv_mean'] >= results['minmax']['cv_mean']:
        best_scaler = 'standard'
    else:
        best_scaler = 'minmax'
    
    print(f"\n  BEST SCALER: {best_scaler.upper()}")
    print(f"  Standard: {results['standard']['cv_mean']:.4f} | MinMax: {results['minmax']['cv_mean']:.4f}")
    
    return {
        'comparison': results,
        'best_scaler': best_scaler
    }


# =============================================================================
# Results Compilation
# =============================================================================
def compile_results(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compile all results into a comparison table.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary with all model results.
    
    Returns
    -------
    pd.DataFrame
        Comparison table sorted by accuracy.
    """
    data = []
    for name, res in all_results.items():
        row = {
            'Model': res.get('model_name', name),
            'Test Accuracy': res.get('accuracy', 0),
            'CV Mean': res.get('cv_mean', None),
            'CV Std': res.get('cv_std', None),
            'Train Time (s)': res.get('train_time', 0)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Test Accuracy', ascending=False)
    return df


# =============================================================================
# Main Comparison Function
# =============================================================================
def run_trees_knn_comparison(X_train, X_test, y_train, y_test,
                              column_info: Dict, seed: int) -> Dict[str, Any]:
    """
    Run the complete Trees and KNN comparison.
    
    This function orchestrates all training and evaluation:
    1. Dummy baseline
    2. Decision Tree (default + tuned)
    3. KNN scaler comparison
    4. KNN (default with best scaler + tuned)
    5. Results compilation
    
    Parameters
    ----------
    X_train, X_test : Training and test features
    y_train, y_test : Training and test labels
    column_info : Column type information for preprocessing
    seed : Random seed for reproducibility
    
    Returns
    -------
    Dict containing all results and the best model
    """
    print("\n" + "=" * 60)
    print("  TASK 3: DECISION TREES AND KNN COMPARISON")
    print("=" * 60)
    
    # Get cross-validation strategy
    cv = get_inner_cv_strategy(seed)
    
    all_results = {}
    
    # =========================================================================
    # Phase 1: Baseline
    # =========================================================================
    all_results['dummy'] = train_dummy_baseline(
        X_train, y_train, X_test, y_test, column_info, seed
    )
    
    # =========================================================================
    # Phase 2: Decision Tree - Default
    # =========================================================================
    all_results['dt_default'] = train_decision_tree(
        X_train, y_train, X_test, y_test, column_info, seed, tune=False
    )
    
    # =========================================================================
    # Phase 3: KNN Scaler Comparison
    # =========================================================================
    scaler_comparison = compare_scalers(
        X_train, y_train, X_test, y_test, column_info, seed, cv
    )
    best_scaler = scaler_comparison['best_scaler']
    
    # =========================================================================
    # Phase 4: KNN - Default (with best scaler)
    # =========================================================================
    all_results['knn_default'] = train_knn(
        X_train, y_train, X_test, y_test, column_info, seed,
        scaler_type=best_scaler, tune=False
    )
    
    # =========================================================================
    # Phase 5: Hyperparameter Tuning
    # =========================================================================
    print_section("HYPERPARAMETER OPTIMIZATION (GridSearchCV)", char="=")
    
    # Decision Tree - Tuned
    all_results['dt_tuned'] = train_decision_tree(
        X_train, y_train, X_test, y_test, column_info, seed,
        tune=True, cv=cv
    )
    
    # KNN - Tuned (with best scaler)
    all_results['knn_tuned'] = train_knn(
        X_train, y_train, X_test, y_test, column_info, seed,
        scaler_type=best_scaler, tune=True, cv=cv
    )
    
    # =========================================================================
    # Phase 6: Results Compilation
    # =========================================================================
    print_section("RESULTS SUMMARY", char="=")
    
    results_df = compile_results(all_results)
    print("\n")
    print(results_df.to_string(index=False))
    
    # Identify best model
    best_model_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Test Accuracy']
    
    # Find the corresponding key
    best_key = None
    for key, res in all_results.items():
        if res.get('model_name') == best_model_name:
            best_key = key
            break
    
    print(f"\n  BEST MODEL: {best_model_name}")
    print(f"  TEST ACCURACY: {best_accuracy:.4f}")
    
    # HPO Analysis
    print_section("HPO ANALYSIS", char="-")
    
    dt_default_acc = all_results['dt_default']['accuracy']
    dt_tuned_acc = all_results['dt_tuned']['accuracy']
    dt_improvement = dt_tuned_acc - dt_default_acc
    
    knn_default_acc = all_results['knn_default']['accuracy']
    knn_tuned_acc = all_results['knn_tuned']['accuracy']
    knn_improvement = knn_tuned_acc - knn_default_acc
    
    print(f"\n  Decision Tree:")
    print(f"    Default: {dt_default_acc:.4f} | Tuned: {dt_tuned_acc:.4f}")
    print(f"    Improvement: {dt_improvement:+.4f}")
    print(f"    HPO Time: {all_results['dt_tuned']['train_time']:.2f}s")
    
    print(f"\n  KNN:")
    print(f"    Default: {knn_default_acc:.4f} | Tuned: {knn_tuned_acc:.4f}")
    print(f"    Improvement: {knn_improvement:+.4f}")
    print(f"    HPO Time: {all_results['knn_tuned']['train_time']:.2f}s")
    
    return {
        'all_results': all_results,
        'results_table': results_df,
        'best_model_key': best_key,
        'best_pipeline': all_results[best_key]['pipeline'],
        'scaler_comparison': scaler_comparison,
        'hpo_analysis': {
            'dt_improvement': dt_improvement,
            'knn_improvement': knn_improvement
        }
    }


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    import os
    from task2_evaluation import setup_evaluation
    
    # Get the data path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "bank_34.pkl")
    
    print("=" * 60)
    print("Task 3: Trees and KNN Comparison")
    print("=" * 60)
    
    # Setup evaluation (get data splits)
    X_train, X_test, y_train, y_test, column_info = setup_evaluation(data_path, SEED)
    
    # Run comparison
    results = run_trees_knn_comparison(
        X_train, X_test, y_train, y_test, column_info, SEED
    )
    
    print("\n" + "=" * 60)
    print("Task 3 Complete!")
    print("=" * 60)
