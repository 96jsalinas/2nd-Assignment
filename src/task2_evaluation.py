"""
Task 2: Outer and Inner Evaluation Setup

This module handles the data splitting strategy and preprocessing pipelines.
- Outer evaluation: Train/Test split (Holdout method) - 80/20 stratified
- Inner evaluation: StratifiedKFold cross-validation for hyperparameter tuning
- Preprocessing: Full pipeline with ColumnTransformer

Main metric: Accuracy
Secondary metric: Confusion matrices

Uses: pandas, numpy, scikit-learn (pipelines, transformers, model_selection)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# Constants
# =============================================================================
SEED = 100562234
TARGET_COLUMN = "deposit"
TEST_SIZE = 0.2  # 80/20 split
N_FOLDS = 3  # For StratifiedKFold


# =============================================================================
# Custom Transformer for pdays
# =============================================================================
class PdaysTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for the 'pdays' variable.
    
    Implements Option C (Two-Feature Approach):
    - Creates a binary indicator: was_contacted_before (1 if pdays != -1, else 0)
    - Transforms pdays: replaces -1 with 0 (no prior contact = 0 days since contact)
    
    This preserves maximum information from the original variable while handling
    the special -1 encoding.
    """
    
    def fit(self, X, y=None):
        """
        Fit method (no computation needed, transformation is stateless).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            The pdays column.
        y : ignored
        
        Returns
        -------
        self
        """
        return self
    
    def transform(self, X):
        """
        Transform pdays to two features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            The pdays column.
        
        Returns
        -------
        X_transformed : array of shape (n_samples, 2)
            Column 0: was_contacted_before (binary)
            Column 1: pdays_transformed (continuous, -1 replaced with 0)
        """
        # Convert to numpy array if DataFrame
        if hasattr(X, 'values'):
            X = X.values
        pdays = X.ravel() if X.ndim > 1 else X
        
        # Binary indicator: was contacted before
        was_contacted = (pdays != -1).astype(int)
        
        # Transform pdays: replace -1 with 0 (no contact = 0 days since contact)
        pdays_transformed = np.where(pdays == -1, 0, pdays)
        
        # Return as 2D array with 2 columns
        return np.column_stack([was_contacted, pdays_transformed])
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for the transformed output."""
        return np.array(['was_contacted_before', 'pdays_transformed'])


# =============================================================================
# Data Loading and Preparation
# =============================================================================
def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Load data and separate features from target.
    
    Parameters
    ----------
    data_path : str
        Path to the pickle file containing the dataset.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]
        X: Feature dataframe
        y: Target series
        column_info: Dictionary with 'numerical', 'categorical', 'pdays' column lists
    """
    # Load data
    df = pd.read_pickle(data_path)
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Identify column types
    # pdays is treated separately due to special preprocessing
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove pdays from numerical (it gets special treatment)
    pdays_col = ['pdays'] if 'pdays' in numerical_cols else []
    numerical_cols = [col for col in numerical_cols if col != 'pdays']
    
    column_info = {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'pdays': pdays_col
    }
    
    return X, y, column_info


# =============================================================================
# Evaluation Setup
# =============================================================================
def setup_evaluation(data_path: str, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                  pd.Series, pd.Series, 
                                                                  Dict[str, List[str]]]:
    """
    Set up the outer evaluation strategy with train/test split.
    
    Uses stratified sampling to maintain class proportions in both sets.
    
    Parameters
    ----------
    data_path : str
        Path to the pickle file containing the dataset.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple containing:
        X_train : pd.DataFrame - Training features
        X_test : pd.DataFrame - Test features
        y_train : pd.Series - Training labels
        y_test : pd.Series - Test labels
        column_info : Dict - Column type information for preprocessing
    """
    # Load and prepare data
    X, y, column_info = load_and_prepare_data(data_path)
    
    # Encode target labels ('yes'/'no' -> 1/0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_encoded = pd.Series(y_encoded, index=y.index)
    
    # Store label encoder classes for reference
    # Classes will be like ['no', 'yes'] -> [0, 1]
    print(f"\n  Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Perform stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y_encoded  # Maintain class proportions
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("OUTER EVALUATION SETUP: Train/Test Split")
    print(f"{'='*60}")
    print(f"  Total samples:     {len(X):,}")
    print(f"  Training samples:  {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test samples:      {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Random seed:       {seed}")
    print(f"\n  Class distribution in training set:")
    train_dist = y_train.value_counts()
    for class_val in train_dist.index:
        print(f"    - {class_val}: {train_dist[class_val]:,} ({train_dist[class_val]/len(y_train)*100:.2f}%)")
    print(f"\n  Class distribution in test set:")
    test_dist = y_test.value_counts()
    for class_val in test_dist.index:
        print(f"    - {class_val}: {test_dist[class_val]:,} ({test_dist[class_val]/len(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, column_info


# =============================================================================
# Inner Cross-Validation Strategy
# =============================================================================
def get_inner_cv_strategy(seed: int = SEED, n_splits: int = N_FOLDS) -> StratifiedKFold:
    """
    Define the inner cross-validation strategy for hyperparameter tuning.
    
    Uses StratifiedKFold to maintain class proportions in each fold,
    which is appropriate for classification problems.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_splits : int
        Number of folds for cross-validation.
    
    Returns
    -------
    StratifiedKFold
        Cross-validation splitter object for use with GridSearchCV/RandomizedSearchCV.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    print(f"\n{'='*60}")
    print("INNER EVALUATION SETUP: Cross-Validation Strategy")
    print(f"{'='*60}")
    print(f"  Strategy:    StratifiedKFold")
    print(f"  Folds:       {n_splits}")
    print(f"  Shuffle:     True")
    print(f"  Random seed: {seed}")
    
    return cv


# =============================================================================
# Preprocessing Pipelines
# =============================================================================
def create_preprocessing_pipeline(column_info: Dict[str, List[str]], 
                                   scaler_type: str = "standard") -> ColumnTransformer:
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    This pipeline handles:
    - Numerical features: Imputation (median) + Scaling
    - Categorical features: Imputation (most frequent) + One-hot encoding
    - pdays: Custom transformation (binary flag + transformed continuous)
    
    Parameters
    ----------
    column_info : Dict[str, List[str]]
        Dictionary with 'numerical', 'categorical', 'pdays' column lists.
    scaler_type : str
        Type of scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
    
    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline ready to be used in a full ML pipeline.
    """
    # Select scaler
    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    elif scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Numerical pipeline: Impute with median, then scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])
    
    # Categorical pipeline: Impute with most frequent, then one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # pdays pipeline: Custom transformer
    pdays_pipeline = Pipeline(steps=[
        ('pdays_transformer', PdaysTransformer()),
        ('scaler', StandardScaler())  # Scale the transformed pdays features
    ])
    
    # Build transformers list
    transformers = []
    
    if column_info['numerical']:
        transformers.append(('num', numerical_pipeline, column_info['numerical']))
    
    if column_info['categorical']:
        transformers.append(('cat', categorical_pipeline, column_info['categorical']))
    
    if column_info['pdays']:
        transformers.append(('pdays', pdays_pipeline, column_info['pdays']))
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def create_full_pipeline(column_info: Dict[str, List[str]], 
                         estimator,
                         scaler_type: str = "standard") -> Pipeline:
    """
    Create a full ML pipeline with preprocessing and estimator.
    
    Parameters
    ----------
    column_info : Dict[str, List[str]]
        Dictionary with column type information.
    estimator : sklearn estimator
        The classifier to use (e.g., KNeighborsClassifier, DecisionTreeClassifier).
    scaler_type : str
        Type of scaler for numerical features.
    
    Returns
    -------
    Pipeline
        Complete pipeline ready for training and prediction.
    """
    preprocessor = create_preprocessing_pipeline(column_info, scaler_type)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])
    
    return pipeline


def print_pipeline_summary(column_info: Dict[str, List[str]], scaler_type: str = "standard"):
    """
    Print a summary of the preprocessing pipeline.
    
    Parameters
    ----------
    column_info : Dict[str, List[str]]
        Dictionary with column type information.
    scaler_type : str
        Type of scaler being used.
    """
    print(f"\n{'='*60}")
    print("PREPROCESSING PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n  Numerical features ({len(column_info['numerical'])}):")
    print(f"    - Imputation: median")
    print(f"    - Scaling: {scaler_type}")
    for col in column_info['numerical']:
        print(f"      • {col}")
    
    print(f"\n  Categorical features ({len(column_info['categorical'])}):")
    print(f"    - Imputation: most frequent")
    print(f"    - Encoding: one-hot")
    for col in column_info['categorical']:
        print(f"      • {col}")
    
    print(f"\n  Special feature: pdays")
    print(f"    - Custom PdaysTransformer (Option C)")
    print(f"    - Output: was_contacted_before (binary) + pdays_transformed (continuous)")
    print(f"    - Scaling: standard (applied after transformation)")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    import os
    
    # Get the data path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "bank_34.pkl")
    
    print("=" * 60)
    print("Task 2: Outer and Inner Evaluation Setup")
    print("=" * 60)
    
    # Setup evaluation (train/test split)
    X_train, X_test, y_train, y_test, column_info = setup_evaluation(data_path, SEED)
    
    # Get inner CV strategy
    cv = get_inner_cv_strategy(SEED)
    
    # Print pipeline summary
    print_pipeline_summary(column_info)
    
    # Test preprocessing pipeline
    print(f"\n{'='*60}")
    print("TESTING PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    
    preprocessor = create_preprocessing_pipeline(column_info, scaler_type="standard")
    
    # Fit and transform training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"\n  Original training shape:    {X_train.shape}")
    print(f"  Transformed training shape: {X_train_transformed.shape}")
    print(f"  Original test shape:        {X_test.shape}")
    print(f"  Transformed test shape:     {X_test_transformed.shape}")
    
    print(f"\n{'='*60}")
    print("Task 2 Setup Complete!")
    print(f"{'='*60}")
