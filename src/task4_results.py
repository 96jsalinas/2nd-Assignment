"""
Task 4: Results and Final Model

This module handles:
- Reporting results from inner evaluation (already done in Task 3)
- Selecting the best model alternative (Decision Tree Tuned)
- Evaluating on test set (outer evaluation - recap)
- Training final model on full training data
- Generating competition predictions

Output: CSV file with competition predictions
"""

import pandas as pd
import numpy as np
import os
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix

# Import from other tasks
from task2_evaluation import (
    create_full_pipeline, 
    load_and_prepare_data,
    SEED
)

def run_final_evaluation(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series, 
                         column_info: dict,
                         task3_results: dict, 
                         seed: int, 
                         output_dir: str) -> dict:
    """
    Run the final evaluation and generate predictions.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test features.
    y_train, y_test : pd.Series
        Training and test labels (numeric encoded).
    column_info : dict
        Column type information.
    task3_results : dict
        Results dictionary from Task 3.
    seed : int
        Random seed.
    output_dir : str
        Directory to save outputs.
    
    Returns
    -------
    dict
        Dictionary containing final results and model information.
    """
    print("\n" + "=" * 60)
    print("  TASK 4: RESULTS AND FINAL MODEL")
    print("=" * 60)

    # 1. Select Best Model
    # Based on Task 3 analysis: Decision Tree Tuned is selected
    # Reasons: Tied accuracy with KNN (82.09%) but significantly faster/simpler.
    print("\n1. BEST MODEL SELECTION")
    print("-" * 30)
    
    # Extract results dictionary
    results_dict = task3_results.get('all_results', task3_results)
    
    best_model_name = "Decision Tree (Tuned)"
    best_pipeline = None
    best_params = None
    
    if 'dt_tuned' in results_dict:
        dt_results = results_dict['dt_tuned']
        best_pipeline = dt_results['pipeline']
        best_params = dt_results.get('best_params', {})
        test_acc = dt_results['accuracy']
        
        print(f"  Selected Model: {best_model_name}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Best Params:    {best_params}")
    else:
        print("  WARNING: 'dt_tuned' not found in results. Using default Decision Tree.")
        from sklearn.tree import DecisionTreeClassifier
        best_model_name = "Decision Tree (Default)"
        dt = DecisionTreeClassifier(random_state=seed)
        best_pipeline = create_full_pipeline(column_info, dt, scaler_type="standard")
    
    # 2. Train Final Model on Full Data
    print("\n2. TRAINING FINAL MODEL ON FULL DATASET")
    print("-" * 30)
    
    # Combine train and test sets
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    
    print(f"  Full dataset shape: {X_full.shape}")
    print(f"  Training final model...")
    
    # Clone the pipeline to reset it, then set params if available
    final_model = clone(best_pipeline)
    
    # Fit on full data
    final_model.fit(X_full, y_full)
    print("  Model trained successfully.")
    
    # Save the model
    # Assignment req: save as joblib
    import joblib
    model_path = os.path.join(output_dir, "final_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"  Model saved to: {model_path}")
    
    # 3. Generate Decision Tree Visualization
    print("\n3. GENERATING DECISION TREE VISUALIZATION")
    print("-" * 30)
    _generate_tree_visualization(final_model, output_dir)
    
    # 4. Generate Competition Predictions
    print("\n4. GENERATING COMPETITION PREDICTIONS")
    print("-" * 30)
    
    # Load competition data
    # Assuming standard project structure: src/../data/bank_competition.pkl
    script_dir = os.path.dirname(os.path.abspath(__file__))
    comp_data_path = os.path.join(script_dir, "..", "data", "bank_competition.pkl")
    
    pred_path = None
    if os.path.exists(comp_data_path):
        print("  Loading competition data...")
        X_comp = pd.read_pickle(comp_data_path)
        print(f"  Competition data shape: {X_comp.shape}")
        
        # Predict (numeric 0/1)
        y_comp_pred_num = final_model.predict(X_comp)
        
        # Map back to strings ('no'/'yes')
        label_map = {0: 'no', 1: 'yes'}
        y_comp_pred_str = [label_map[y] for y in y_comp_pred_num]
        
        # Create a display sample with raw features for verification
        display_cols = ['age', 'job', 'marital', 'balance', 'pdays']
        display_df = X_comp[display_cols].copy()
        display_df['Prediction'] = y_comp_pred_str
        
        # Create submission DataFrame (standard format)
        submission_df = pd.DataFrame({
            'Id': X_comp.index,
            'deposit': y_comp_pred_str
        })
        
        # Save to CSV
        pred_path = os.path.join(output_dir, "competition_predictions.csv")
        submission_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved to: {pred_path}")
        print("\n  Sample predictions with raw features:")
        print(display_df.head())
        
    else:
        print(f"  ERROR: Competition file not found at {comp_data_path}")
    
    return {
        'best_model_name': best_model_name,
        'final_model': final_model,
        'model_path': model_path,
        'predictions_path': pred_path
    }


def _generate_tree_visualization(final_model, output_dir: str):
    """
    Generate a visual representation of the decision tree for the report.
    
    Parameters
    ----------
    final_model : Pipeline
        The trained pipeline containing the DecisionTreeClassifier.
    output_dir : str
        Directory to save outputs.
    """
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    
    # Create figures directory in report folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "report", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Extract the DecisionTreeClassifier from the pipeline
    tree_clf = final_model.named_steps['classifier']
    
    # Get feature names from the preprocessor
    preprocessor = final_model.named_steps['preprocessor']
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        print(f"  Could not get feature names: {e}")
        feature_names = None
    
    # Class names
    class_names = ['no', 'yes']
    
    # Create simplified visualization (top 3 levels, only splitting rules)
    fig, ax = plt.subplots(figsize=(24, 12))  # Increased width for depth 3
    
    # Plot standard tree first
    plot_tree(
        tree_clf, 
        feature_names=feature_names, 
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=10,
        max_depth=3,  # Increased depth
        impurity=False, # Hide gini
        node_ids=False,
        proportion=False
    )
    
    # Post-process text labels to keep only splitting rule
    # The splitting rule is always the first line of text in non-leaf nodes
    for text in ax.texts:
        label = text.get_text()
        if "<=" in label:  # It's a splitting node
            # Keep only the first line (the condition)
            text.set_text(label.split("\n")[0])
        else:
            # For leaf nodes, show simple class or nothing
            # Let's show the class prediction for context if it's a leaf
            if "class =" in label:
                # Extract just "class = yes/no"
                lines = label.split("\n")
                for line in lines:
                    if "class =" in line:
                        text.set_text(line)
                        break
    
    plt.title("Decision Tree Decision Rules (Top 3 Levels)", fontsize=16)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, "decision_tree_simple.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Tree visualization saved to: {fig_path}")
    print(f"  Tree depth: {tree_clf.get_depth()}, Leaves: {tree_clf.get_n_leaves()}")

if __name__ == "__main__":
    # For standalone testing
    print("Task 4 module")
