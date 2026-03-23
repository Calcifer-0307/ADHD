from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from get_data import get_data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set global random seed for reproducibility
np.random.seed(42)

# 1. Data Preparation
X, y1, y2 = get_data()
y = np.column_stack((y1, y2)) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data shape: X={X.shape}, y={y.shape}")
print("="*50)

# 2. Define Models and Hyperparameter Grids
models_config = {
    "RandomForest": {
        "base_estimator": RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
        "param_grid": {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [10, 20],
            'estimator__min_samples_split': [2, 5]
        }
    },
    "XGBoost": {
        "base_estimator": XGBClassifier(random_state=42, eval_metric='logloss'),
        "param_grid": {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [3, 5, 7],
            'estimator__learning_rate': [0.01, 0.1]
        }
    },
    "SVC": {
        "base_estimator": SVC(random_state=42, class_weight='balanced'),
        "param_grid": {
            'estimator__C': [0.1, 1, 10],
            'estimator__kernel': ['rbf', 'linear']
        }
    }
}

# 3. Evaluation Function
def evaluate_label(y_true, y_pred, label_name, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Label [{label_name}]:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - F1 Score (Positive Class): {f1_pos:.4f}")
    print(f"  - F1 Score (Macro): {f1_mac:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    
    if label_name == "ADHD_Outcome":
        labels = ["Non-ADHD (0)", "ADHD (1)"]
    elif label_name == "Sex_F":
        labels = ["Male (0)", "Female (1)"]
    else:
        labels = ["0", "1"]
        
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} - {label_name} Confusion Matrix")
    
    os.makedirs("reports/Confusion_Matrix", exist_ok=True)
    save_path = os.path.join("reports/Confusion_Matrix", f"{model_name}_{label_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  - Confusion matrix saved to: {save_path}")

# 4. Helper Function for Markdown Reporting
def generate_markdown_report(model_name, best_params, best_cv_score, y_true, y_pred, overall_acc):
    """
    Generates a formatted markdown string containing the evaluation metrics for a specific model.
    """
    report = f"## {model_name}\n\n"
    report += f"- **Best Parameters:** `{best_params}`\n"
    report += f"- **Best CV F1 Score (Macro):** `{best_cv_score:.4f}`\n\n"
    
    # Evaluate ADHD_Outcome (Label 0)
    acc_0 = accuracy_score(y_true[:, 0], y_pred[:, 0])
    f1_pos_0 = f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0)
    f1_mac_0 = f1_score(y_true[:, 0], y_pred[:, 0], average='macro', zero_division=0)
    
    report += "### Label: ADHD_Outcome\n"
    report += f"- Accuracy: `{acc_0:.4f}`\n"
    report += f"- F1 Score (Positive Class): `{f1_pos_0:.4f}`\n"
    report += f"- F1 Score (Macro): `{f1_mac_0:.4f}`\n\n"
    
    # Evaluate Sex_F (Label 1)
    acc_1 = accuracy_score(y_true[:, 1], y_pred[:, 1])
    f1_pos_1 = f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0)
    f1_mac_1 = f1_score(y_true[:, 1], y_pred[:, 1], average='macro', zero_division=0)
    
    report += "### Label: Sex_F\n"
    report += f"- Accuracy: `{acc_1:.4f}`\n"
    report += f"- F1 Score (Positive Class): `{f1_pos_1:.4f}`\n"
    report += f"- F1 Score (Macro): `{f1_mac_1:.4f}`\n\n"
    
    # Overall Accuracy
    report += f"**Overall Accuracy (Subset Accuracy):** `{overall_acc:.4f}`\n\n"
    report += "---\n\n"
    
    return report

# 5. Training and Evaluation Loop
# Define a cross-validation strategy with a fixed random state
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the markdown string to store all results
md_report = "# MultiOutput Classification Results\n\n"
md_report += f"**Data shape:** X={X.shape}, y={y.shape}\n\n"

for model_name, config in models_config.items():
    print(f"\n{'='*20} Training Model: {model_name} {'='*20}")
    
    # 5.1 Initialize and Train Model
    multi_output_clf = MultiOutputClassifier(config["base_estimator"], n_jobs=-1)
    grid_search = GridSearchCV(
        multi_output_clf, 
        config["param_grid"], 
        cv=cv_strategy, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # 5.2 Print CV Results to Terminal
    print(f"[{model_name}] Best Parameters:", grid_search.best_params_)
    print(f"[{model_name}] Best CV F1 Score (Macro): {grid_search.best_score_:.4f}")
    print("-" * 30)
    
    # 5.3 Make Predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 5.4 Evaluate and Print to Terminal (also saves Confusion Matrix images)
    evaluate_label(y_test[:, 0], y_pred[:, 0], "ADHD_Outcome", model_name)
    print("-" * 30)
    evaluate_label(y_test[:, 1], y_pred[:, 1], "Sex_F", model_name)
    
    overall_acc = (y_pred == y_test).all(axis=1).mean()
    print("-" * 30)
    print(f"[{model_name}] Overall Accuracy (Subset Accuracy): {overall_acc:.4f}")
    print("="*60)
    
    # 5.5 Append Model Results to Markdown Report
    md_report += generate_markdown_report(
        model_name=model_name,
        best_params=grid_search.best_params_,
        best_cv_score=grid_search.best_score_,
        y_true=y_test,
        y_pred=y_pred,
        overall_acc=overall_acc
    )

# 6. Save the Final Markdown Report
os.makedirs("reports", exist_ok=True)
report_path = "reports/MultiOutput_Results.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md_report)
print(f"All results have been saved to {report_path}")