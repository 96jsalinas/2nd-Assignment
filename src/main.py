"""
Main entry point for the Advanced Programming 2nd Assignment.
Bank Marketing Dataset Analysis - Classification Task

Student ID: 100562234 (used as random seed)
Python Version: 3.13.5 (Conda)
IDE: Positron

This script orchestrates the execution of all tasks defined in the requirements.
"""

import time
import os

# Import task modules
from task1_eda import run_eda
from task2_evaluation import setup_evaluation
from task3_trees_knn import run_trees_knn_comparison
from task4_results import run_final_evaluation
from task5_open_choice import run_catboost_task
from task6_feature_selection import run_feature_selection

# Constants
SEED = 100562234
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../data/bank_34.pkl")
COMPETITION_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/bank_competition.pkl")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../outputs/")


def main():
    """
    Main function to execute all tasks sequentially.
    """
    print("=" * 60)
    print("Advanced Programming - 2nd Assignment")
    print("Bank Marketing Dataset Analysis")
    print("=" * 60)
    
    total_start = time.time()
    
    # Task 1: Simplified EDA
    print("\n" + "=" * 60)
    print("TASK 1: Simplified Exploratory Data Analysis")
    print("=" * 60)
    # run_eda(DATA_PATH)  # Uncomment to run EDA
    eda_results = run_eda(DATA_PATH)
    
    # Task 2: Outer and Inner Evaluation Setup
    print("\n" + "=" * 60)
    print("TASK 2: Outer and Inner Evaluation Setup")
    print("=" * 60)
    X_train, X_test, y_train, y_test, column_info = setup_evaluation(DATA_PATH, SEED)
    
    # Task 3: Trees and KNN Comparison
    print("\n" + "=" * 60)
    print("TASK 3: Decision Trees and KNN Comparison")
    print("=" * 60)
    task3_results = run_trees_knn_comparison(X_train, X_test, y_train, y_test, column_info, SEED)
    
    # Task 4: Results and Final Model
    print("\n" + "=" * 60)
    print("TASK 4: Results and Final Model")
    print("=" * 60)
    run_final_evaluation(X_train, X_test, y_train, y_test, column_info, 
                         task3_results, SEED, OUTPUT_PATH)
    
    # Task 5: Open-Choice Task
    print("\n" + "=" * 60)
    print("TASK 5: Open-Choice Task")
    print("=" * 60)
    run_catboost_task(DATA_PATH)
    
    # Task 6: Feature Selection (Optional)
    print("\n" + "=" * 60)
    print("TASK 6: Feature Selection for KNN (Optional)")
    print("=" * 60)
    run_feature_selection(X_train, X_test, y_train, y_test, column_info, SEED)
    
    total_end = time.time()
    
    print("\n" + "=" * 60)
    print(f"All tasks completed in {total_end - total_start:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
