"""
Task 1: Simplified Exploratory Data Analysis (EDA)

This module performs a simplified EDA on the Bank Marketing dataset.
Focus areas:
- Number of features and instances
- Categorical vs. numerical variables identification
- High cardinality categorical variables
- Missing values analysis
- Constant columns and ID columns detection
- Problem type identification (classification)
- Class imbalance check
- Special attention to 'pdays' variable preprocessing

Response Variable: deposit

Uses: pandas, numpy, scikit-learn (where applicable)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


# =============================================================================
# Constants
# =============================================================================
TARGET_COLUMN = "deposit"
HIGH_CARDINALITY_THRESHOLD = 10  # Threshold for considering a variable high cardinality


# =============================================================================
# Main EDA Function
# =============================================================================
def run_eda(data_path: str) -> Dict[str, Any]:
    """
    Perform simplified exploratory data analysis.
    
    This function orchestrates all EDA steps and prints formatted results
    to the console while returning a structured dictionary with all findings.
    
    Parameters
    ----------
    data_path : str
        Path to the pickle file containing the dataset.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all EDA results and insights.
    """
    # Load data
    print("Loading data...")
    df = pd.read_pickle(data_path)
    print(f"Data loaded successfully from: {data_path}\n")
    
    results = {}
    
    # 1. Dataset Overview
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    results["overview"] = get_dataset_overview(df)
    print(f"  Number of instances: {results['overview']['n_instances']:,}")
    print(f"  Number of features:  {results['overview']['n_features']}")
    print(f"  Total columns:       {results['overview']['n_columns']}")
    print(f"  Target variable:     {TARGET_COLUMN}")
    
    # 2. Variable Classification
    print("\n" + "=" * 60)
    print("2. VARIABLE CLASSIFICATION")
    print("=" * 60)
    results["variables"] = classify_variables(df)
    print(f"\n  Numerical variables ({len(results['variables']['numerical'])}):")
    for var in results["variables"]["numerical"]:
        print(f"    - {var}")
    print(f"\n  Categorical variables ({len(results['variables']['categorical'])}):")
    for var in results["variables"]["categorical"]:
        print(f"    - {var}")
    
    # 3. High Cardinality Check
    print("\n" + "=" * 60)
    print("3. HIGH CARDINALITY CATEGORICAL VARIABLES")
    print("=" * 60)
    results["cardinality"] = check_high_cardinality(df, threshold=HIGH_CARDINALITY_THRESHOLD)
    print(f"\n  Threshold used: > {HIGH_CARDINALITY_THRESHOLD} unique values")
    if results["cardinality"]["high_cardinality_vars"]:
        print(f"\n  High cardinality variables found:")
        for var, count in results["cardinality"]["high_cardinality_vars"].items():
            print(f"    - {var}: {count} unique values")
    else:
        print("\n  No high cardinality categorical variables found.")
    print(f"\n  All categorical variable cardinalities:")
    for var, count in results["cardinality"]["all_cardinalities"].items():
        print(f"    - {var}: {count} unique values")
    
    # 4. Missing Values
    print("\n" + "=" * 60)
    print("4. MISSING VALUES ANALYSIS")
    print("=" * 60)
    results["missing"] = analyze_missing_values(df)
    if results["missing"]["columns_with_missing"]:
        print(f"\n  Columns with missing values:")
        for col, info in results["missing"]["columns_with_missing"].items():
            print(f"    - {col}: {info['count']:,} missing ({info['percentage']:.2f}%)")
    else:
        print("\n  No missing values found in the dataset.")
    print(f"\n  Total missing values: {results['missing']['total_missing']:,}")
    
    # 5. Constant and ID Columns
    print("\n" + "=" * 60)
    print("5. CONSTANT AND ID COLUMNS")
    print("=" * 60)
    results["constant_id"] = check_constant_id_columns(df)
    if results["constant_id"]["constant_columns"]:
        print(f"\n  Constant columns found:")
        for col in results["constant_id"]["constant_columns"]:
            print(f"    - {col}")
    else:
        print("\n  No constant columns found.")
    if results["constant_id"]["potential_id_columns"]:
        print(f"\n  Potential ID columns (all unique values):")
        for col in results["constant_id"]["potential_id_columns"]:
            print(f"    - {col}")
    else:
        print("\n  No potential ID columns found.")
    
    # 6. Problem Type
    print("\n" + "=" * 60)
    print("6. PROBLEM TYPE IDENTIFICATION")
    print("=" * 60)
    results["problem_type"] = identify_problem_type(df, TARGET_COLUMN)
    print(f"\n  Target variable: {TARGET_COLUMN}")
    print(f"  Target dtype: {results['problem_type']['target_dtype']}")
    print(f"  Unique values: {results['problem_type']['n_unique']}")
    print(f"  Problem type: {results['problem_type']['problem_type'].upper()}")
    
    # 7. Class Imbalance
    print("\n" + "=" * 60)
    print("7. CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    results["imbalance"] = analyze_class_distribution(df, TARGET_COLUMN)
    print(f"\n  Class distribution:")
    for class_val, info in results["imbalance"]["class_counts"].items():
        print(f"    - {class_val}: {info['count']:,} ({info['percentage']:.2f}%)")
    print(f"\n  Imbalance ratio: {results['imbalance']['imbalance_ratio']:.2f}")
    print(f"  Is imbalanced (>1.5 ratio): {results['imbalance']['is_imbalanced']}")
    
    # 8. pdays Analysis
    print("\n" + "=" * 60)
    print("8. SPECIAL VARIABLE ANALYSIS: pdays")
    print("=" * 60)
    results["pdays"] = analyze_pdays(df)
    print(f"\n  Total observations: {results['pdays']['total']:,}")
    print(f"  Values equal to -1 (no previous contact): {results['pdays']['no_contact_count']:,} ({results['pdays']['no_contact_pct']:.2f}%)")
    print(f"  Values > -1 (previous contact): {results['pdays']['contacted_count']:,} ({results['pdays']['contacted_pct']:.2f}%)")
    print(f"\n  Statistics for contacted clients (pdays > -1):")
    print(f"    - Min:    {results['pdays']['valid_stats']['min']}")
    print(f"    - Max:    {results['pdays']['valid_stats']['max']}")
    print(f"    - Mean:   {results['pdays']['valid_stats']['mean']:.2f}")
    print(f"    - Median: {results['pdays']['valid_stats']['median']:.2f}")
    print(f"\n  Preprocessing recommendation: {results['pdays']['recommendation']}")
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    
    return results


# =============================================================================
# Individual EDA Functions
# =============================================================================

def get_dataset_overview(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get basic dataset dimensions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    
    Returns
    -------
    Dict[str, int]
        Dictionary with n_instances, n_features, n_columns.
    """
    n_instances, n_columns = df.shape
    n_features = n_columns - 1  # Excluding target
    
    return {
        "n_instances": n_instances,
        "n_features": n_features,
        "n_columns": n_columns
    }


def classify_variables(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify variables as numerical or categorical based on dtype.
    
    Uses pandas dtype detection. Object types are considered categorical,
    numeric types (int, float) are considered numerical.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'numerical' and 'categorical' lists.
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    return {
        "numerical": numerical,
        "categorical": categorical
    }


def check_high_cardinality(df: pd.DataFrame, threshold: int = 10) -> Dict[str, Any]:
    """
    Check for high cardinality categorical variables.
    
    A categorical variable is considered high cardinality if it has more
    unique values than the specified threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    threshold : int
        Number of unique values above which a variable is high cardinality.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with high_cardinality_vars and all_cardinalities.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    
    all_cardinalities = {}
    high_cardinality_vars = {}
    
    for col in categorical_cols:
        n_unique = df[col].nunique()
        all_cardinalities[col] = n_unique
        if n_unique > threshold:
            high_cardinality_vars[col] = n_unique
    
    return {
        "threshold": threshold,
        "high_cardinality_vars": high_cardinality_vars,
        "all_cardinalities": all_cardinalities
    }


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with missing value counts and percentages.
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    n_rows = len(df)
    
    columns_with_missing = {}
    for col in df.columns:
        if missing_counts[col] > 0:
            columns_with_missing[col] = {
                "count": int(missing_counts[col]),
                "percentage": (missing_counts[col] / n_rows) * 100
            }
    
    return {
        "total_missing": int(total_missing),
        "columns_with_missing": columns_with_missing,
        "n_columns_with_missing": len(columns_with_missing)
    }


def check_constant_id_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check for constant columns (all same value) and potential ID columns
    (all unique values).
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with constant_columns and potential_id_columns.
    """
    constant_columns = []
    potential_id_columns = []
    
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique == 1:
            constant_columns.append(col)
        elif n_unique == len(df):
            potential_id_columns.append(col)
    
    return {
        "constant_columns": constant_columns,
        "potential_id_columns": potential_id_columns
    }


def identify_problem_type(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Identify if this is a regression or classification problem.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    target : str
        Name of the target column.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with target info and problem type.
    """
    target_series = df[target]
    target_dtype = str(target_series.dtype)
    n_unique = target_series.nunique()
    
    # Classification if categorical or few unique values
    if target_series.dtype == "object" or n_unique <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"
    
    return {
        "target_column": target,
        "target_dtype": target_dtype,
        "n_unique": n_unique,
        "unique_values": target_series.unique().tolist(),
        "problem_type": problem_type
    }


def analyze_class_distribution(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Analyze class distribution for classification problems.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    target : str
        Name of the target column.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with class counts, percentages, and imbalance info.
    """
    value_counts = df[target].value_counts()
    n_total = len(df)
    
    class_counts = {}
    for class_val in value_counts.index:
        count = value_counts[class_val]
        class_counts[class_val] = {
            "count": int(count),
            "percentage": (count / n_total) * 100
        }
    
    # Calculate imbalance ratio (majority / minority)
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    
    # Consider imbalanced if ratio > 1.5
    is_imbalanced = imbalance_ratio > 1.5
    
    return {
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": is_imbalanced
    }


def analyze_pdays(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Special analysis for the 'pdays' variable.
    
    The 'pdays' variable represents the number of days since the client
    was last contacted from a previous campaign. Value -1 indicates
    no/unknown previous contact.
    
    Preprocessing Strategy (Option C - Recommended):
    - Create binary indicator: was_contacted_before = (pdays != -1)
    - For continuous pdays: replace -1 with a large value or use as-is with the flag
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    
    Returns
    -------
    Dict[str, Any]
        Analysis results and preprocessing recommendations for pdays.
    """
    pdays = df["pdays"]
    total = len(pdays)
    
    # Count -1 values (no previous contact)
    no_contact_count = (pdays == -1).sum()
    contacted_count = (pdays != -1).sum()
    
    no_contact_pct = (no_contact_count / total) * 100
    contacted_pct = (contacted_count / total) * 100
    
    # Statistics for valid pdays (where contact occurred)
    valid_pdays = pdays[pdays != -1]
    valid_stats = {
        "min": int(valid_pdays.min()) if len(valid_pdays) > 0 else None,
        "max": int(valid_pdays.max()) if len(valid_pdays) > 0 else None,
        "mean": float(valid_pdays.mean()) if len(valid_pdays) > 0 else None,
        "median": float(valid_pdays.median()) if len(valid_pdays) > 0 else None,
        "std": float(valid_pdays.std()) if len(valid_pdays) > 0 else None
    }
    
    recommendation = (
        "Option C (Two-Feature Approach): Create a binary indicator "
        "'was_contacted_before' (1 if pdays != -1, 0 otherwise) and keep a "
        "transformed 'pdays' column. This preserves maximum information from "
        "the original variable while handling the special -1 encoding."
    )
    
    return {
        "total": total,
        "no_contact_count": int(no_contact_count),
        "no_contact_pct": no_contact_pct,
        "contacted_count": int(contacted_count),
        "contacted_pct": contacted_pct,
        "valid_stats": valid_stats,
        "recommendation": recommendation
    }


# =============================================================================
# Utility Functions for Export
# =============================================================================

def export_eda_summary_to_dict(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export EDA results in a format suitable for LaTeX table generation.
    
    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary from run_eda.
    
    Returns
    -------
    Dict[str, Any]
        Formatted dictionary for export.
    """
    return {
        "instances": results["overview"]["n_instances"],
        "features": results["overview"]["n_features"],
        "numerical_vars": len(results["variables"]["numerical"]),
        "categorical_vars": len(results["variables"]["categorical"]),
        "missing_columns": results["missing"]["n_columns_with_missing"],
        "total_missing": results["missing"]["total_missing"],
        "is_classification": results["problem_type"]["problem_type"] == "classification",
        "is_imbalanced": results["imbalance"]["is_imbalanced"],
        "imbalance_ratio": results["imbalance"]["imbalance_ratio"],
        "pdays_no_contact_pct": results["pdays"]["no_contact_pct"]
    }


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Run EDA on the bank dataset
    import os
    
    # Get the data path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "bank_34.pkl")
    
    # Run EDA
    results = run_eda(data_path)
    
    # Print summary for LaTeX
    print("\n" + "=" * 60)
    print("SUMMARY FOR LATEX EXPORT")
    print("=" * 60)
    summary = export_eda_summary_to_dict(results)
    for key, value in summary.items():
        print(f"  {key}: {value}")
