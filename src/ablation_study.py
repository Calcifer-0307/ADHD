from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score
from get_data import get_data
import numpy as np

# Set global random seed for reproducibility
np.random.seed(42)

# 1. Data Preparation using existing get_data()
X_with_sex, y_adhd, _ = get_data(keep_sex=True)

# 2. Identify and remove Sex_F feature for ablation
# In get_data.py, Sex_F is the second column in cleaned_train_categorical_ohe.csv
# So it's at index 0 of the categorical features slice, which means index 0 of X.
X_without_sex = np.delete(X_with_sex, 0, axis=1)

print("="*50)
print("Starting Feature Ablation Study (XGBoost on ADHD_Outcome)")
print("="*50)

# Define Evaluation Function
def train_and_evaluate(X_data, y_data, condition_name):
    print(f"\n--- Condition: {condition_name} ---")
    print(f"Data shape: X={X_data.shape}, y={y_data.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_clf = XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    
    grid_search = GridSearchCV(
        base_clf, 
        param_grid, 
        cv=cv_strategy, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Positive Class): {f1_pos:.4f}")
    print(f"F1 Score (Macro): {f1_mac:.4f}")
    
    return {"acc": acc, "f1_pos": f1_pos, "f1_mac": f1_mac}

# 3. Run Experiments
results_with = train_and_evaluate(X_with_sex, y_adhd, "WITH Sex_F Feature")
results_without = train_and_evaluate(X_without_sex, y_adhd, "WITHOUT Sex_F Feature")

print("\n" + "="*50)
print("Ablation Study Summary:")
print(f"F1 Score (Positive Class) Difference: {results_with['f1_pos'] - results_without['f1_pos']:.4f} (With - Without)")
print(f"F1 Score (Macro) Difference: {results_with['f1_mac'] - results_without['f1_mac']:.4f} (With - Without)")
print("="*50)