from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from get_data import get_data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings to keep terminal output clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set global random seed for reproducibility
np.random.seed(42)

# 1. Data Preparation
from sklearn.preprocessing import StandardScaler

X, y1, y2 = get_data() # y1: ADHD_Outcome, y2: Sex_F

# Apply standard scaling to features (crucial for SVC, Logistic Regression, and Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X_scaled, y1, y2, test_size=0.2, random_state=42
)

print(f"Data shape: X={X.shape}, y1={y1.shape}, y2={y2.shape}")
print("="*50)

# 2. Define Models and Hyperparameter Grids
models_config = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
        "param_grid": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    },
    "XGBoost": {
        "estimator": XGBClassifier(random_state=42, eval_metric='logloss'),
        "param_grid": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    "SVC": {
        "estimator": SVC(random_state=42, class_weight='balanced', max_iter=2000), # 增加 max_iter 防止无限计算
        "param_grid": {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    "LogisticRegression": {
        "estimator": LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000), # 增加迭代次数以解决不收敛
        "param_grid": {
            'C': [0.01, 0.1, 1, 10], # 移除 100 防止过硬的惩罚导致难收敛
            'solver': ['liblinear', 'lbfgs'] # 替换为 lbfgs（对 l2 优化更好，通常比 saga 快）
        }
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "param_grid": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10]
        }
    },
    "NeuralNetwork": {
        "estimator": MLPClassifier(random_state=42, max_iter=2000, early_stopping=True, n_iter_no_change=20),
        "param_grid": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

# 3. Helper Functions
def evaluate_and_plot_cm(y_true, y_pred, label_name, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - F1 Score (Positive Class): {f1_pos:.4f}")
    print(f"  - F1 Score (Macro): {f1_mac:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages relative to the total number of samples
    cm_percentage = cm.astype('float') / cm.sum()
    
    plt.figure(figsize=(6, 4))
    
    if label_name == "ADHD_Outcome":
        labels = ["Non-ADHD (0)", "ADHD (1)"]
    elif label_name == "Sex_F":
        labels = ["Male (0)", "Female (1)"]
    else:
        labels = ["0", "1"]
        
    # Create annotations that show both count and percentage
    annot_data = [[f"{count}\n({percent:.1%})" for count, percent in zip(row_count, row_percent)] 
                  for row_count, row_percent in zip(cm, cm_percentage)]
        
    sns.heatmap(cm_percentage, annot=annot_data, fmt="", cmap="Blues", 
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1) # Set scale from 0 to 1 for percentage
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} - {label_name} Confusion Matrix")
    
    os.makedirs("reports/Confusion_Matrix/SingleLabel", exist_ok=True)
    save_path = os.path.join("reports/Confusion_Matrix/SingleLabel", f"{model_name}_{label_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return acc, f1_pos, f1_mac

def generate_markdown_section(model_name, label_name, best_params, best_cv_score, acc, f1_pos, f1_mac):
    report = f"### Model: {model_name}\n\n"
    report += f"- **Best Parameters:** `{best_params}`\n"
    report += f"- **Best CV F1 Score (Macro):** `{best_cv_score:.4f}`\n"
    report += f"- **Test Accuracy:** `{acc:.4f}`\n"
    report += f"- **Test F1 Score (Positive Class):** `{f1_pos:.4f}`\n"
    report += f"- **Test F1 Score (Macro):** `{f1_mac:.4f}`\n\n"
    return report

# 4. Training and Evaluation Loop
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

md_report = "# Single Label Independent RandomizedSearch Results\n\n"
md_report += f"**Data shape:** X={X.shape}, y_ADHD={y1.shape}, y_Sex={y2.shape}\n\n"

tasks = [
    ("ADHD_Outcome", y1_train, y1_test),
    ("Sex_F", y2_train, y2_test)
]

for label_name, y_train, y_test in tasks:
    print(f"\n{'#'*30} Target Label: {label_name} {'#'*30}")
    md_report += f"## Target Label: {label_name}\n\n"
    
    for model_name, config in models_config.items():
        print(f"\n{'='*15} Training Model: {model_name} {'='*15}")
        
        # 4.1 Initialize and Train Model using RandomizedSearchCV
        search = RandomizedSearchCV(
            config["estimator"], 
            config["param_grid"], 
            n_iter=42,          # 随机尝试 42 组参数，你可以根据需要调整
            cv=cv_strategy, 
            scoring='f1_macro', 
            n_jobs=-1,
            random_state=42     # 保证每次随机搜索的结果可复现
        )
        search.fit(X_train, y_train)
        
        # 4.2 Print CV Results to Terminal
        print(f"Best Parameters: {search.best_params_}")
        print(f"Best CV F1 Score (Macro): {search.best_score_:.4f}")
        
        # 4.3 Make Predictions
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # 4.4 Evaluate and Print to Terminal
        acc, f1_pos, f1_mac = evaluate_and_plot_cm(y_test, y_pred, label_name, model_name)
        
        # 4.5 Append Model Results to Markdown Report
        md_report += generate_markdown_section(
            model_name=model_name,
            label_name=label_name,
            best_params=search.best_params_,
            best_cv_score=search.best_score_,
            acc=acc,
            f1_pos=f1_pos,
            f1_mac=f1_mac
        )
    md_report += "---\n\n"

# 5. Save the Final Markdown Report
os.makedirs("reports", exist_ok=True)
report_path = "reports/SingleLabel_RandomizedSearch_Results.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md_report)
print(f"\nAll results have been saved to {report_path}")
