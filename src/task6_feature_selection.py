"""
Task 6: Feature Selection for KNN (Optional)

This module implements feature selection using SelectKBest and GridSearchCV
to optimize the number of features for the KNN model, using the optimal
hyperparameters found in Task 3.
"""

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from task2_evaluation import create_full_pipeline, SEED

N_JOBS = -1

def run_feature_selection(X_train, X_test, y_train, y_test, column_info: dict, seed: int = SEED):
    """
    Run feature selection for KNN.
    """
    print("\n" + "=" * 60)
    print("  TASK 6: FEATURE SELECTION FOR KNN")
    print("=" * 60)
    
    # 1. Optimal KNN Parameters (from Task 3)
    # k=11, weights='distance', metric='euclidean', scaler='standard'
    best_k = 11
    best_weights = 'distance'
    best_metric = 'euclidean'
    scaler_type = 'standard'
    
    print(f"  Using optimal KNN params: k={best_k}, weights={best_weights}, metric={best_metric}, scaler={scaler_type}")
    
    # 2. Create Base Pipeline (Preprocessing + Scaling) 
    # Base KNN estimator
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, metric=best_metric)
    
    # Full pipeline: Preprocessor -> KNN
    base_pipeline = create_full_pipeline(column_info, knn, scaler_type=scaler_type)
    
    preprocessor_step = base_pipeline.steps[0][1]
    
    # Construct new pipeline with SelectKBest
    selector = SelectKBest(score_func=f_classif)
    
    pipe_selection = Pipeline(steps=[
        ('preprocessor', preprocessor_step),
        ('selector', selector),
        ('classifier', knn)
    ])
    
    # 3. Grid Search for Optimal Number of Features
    # Total features after OHE = 53 (known from Task 2)
    k_range = list(range(5, 54, 2))  # Every 2nd feature count
    if 53 not in k_range:
        k_range.append(53)
    
    param_grid = {
        'selector__k': k_range
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    print(f"  GridSearching optimal features (Range: {min(k_range)}-{max(k_range)})...")
    
    grid = GridSearchCV(
        pipe_selection,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=N_JOBS,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    # 4. Results
    best_k_features = grid.best_params_['selector__k']
    best_score = grid.best_score_
    
    print(f"\n  Best number of features: {best_k_features}")
    print(f"  Best CV Accuracy: {best_score:.4f}")
    
    # 5. Identify Selected Features
    best_pipeline = grid.best_estimator_
    selector_step = best_pipeline.named_steps['selector']
    preprocessor = best_pipeline.named_steps['preprocessor']
    
    # Get feature names from preprocessor
    try:
        feature_names = preprocessor.get_feature_names_out()
        support = selector_step.get_support()
        selected_features = feature_names[support]
        
        print(f"\n  Selected Features ({len(selected_features)}):")
        # Print first 10 and last 10 if too many
        if len(selected_features) > 20:
            print(list(selected_features[:10]) + ["..."] + list(selected_features[-10:]))
        else:
            print(selected_features)
            
    except Exception as e:
        print(f"  Could not extract feature names: {e}")
        selected_features = []
        feature_names = []

    # 6. Evaluate on Test Set
    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy with selected features: {test_acc:.4f}")
        
    return {
        'best_k': best_k_features,
        'test_accuracy': test_acc,
        'num_total_features': len(feature_names),
        'selected_features': selected_features.tolist() if hasattr(selected_features, 'tolist') else selected_features
    }

if __name__ == "__main__":
    import os
    from task2_evaluation import setup_evaluation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_pth = os.path.join(script_dir, "..", "data", "bank_34.pkl")
    X_tr, X_te, y_tr, y_te, c_info = setup_evaluation(data_pth, SEED)
    run_feature_selection(X_tr, X_te, y_tr, y_te, c_info, SEED)
