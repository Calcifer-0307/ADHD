# Single Label RandomizedSearch Summary Table

| Target Label | Model | Best Parameters | Best CV F1 (Macro) | Test Accuracy | Test F1 (Positive) | Test F1 (Negative) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ADHD_Outcome | RandomForest | `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 10}` | 0.7320 | 0.7810 | 0.8499 | 0.5954 |
| ADHD_Outcome | XGBoost | `{'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 1.0}` | 0.7466 | 0.8058 | 0.8669 | 0.6412 |
| ADHD_Outcome | SVC | `{'kernel': 'rbf', 'gamma': 'scale', 'C': 1}` | 0.7046 | 0.7149 | 0.7928 | 0.5430 |
| ADHD_Outcome | LogisticRegression | `{'solver': 'lbfgs', 'C': 0.01}` | 0.7219 | 0.7479 | 0.8076 | 0.6347 |
| ADHD_Outcome | DecisionTree | `{'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 5, 'criterion': 'entropy'}` | 0.6957 | 0.7603 | 0.8199 | 0.6420 |
| ADHD_Outcome | NeuralNetwork | `{'solver': 'adam', 'learning_rate_init': 0.01, 'hidden_layer_sizes': (100,), 'alpha': 0.001, 'activation': 'relu'}` | 0.6805 | 0.7438 | 0.8218 | 0.5441 |
| Sex_F | RandomForest | `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10}` | 0.5653 | 0.6942 | 0.3148 | 0.8032 |
| Sex_F | XGBoost | `{'subsample': 1.0, 'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.8}` | 0.7069 | 0.8058 | 0.6667 | 0.8630 |
| Sex_F | SVC | `{'kernel': 'rbf', 'gamma': 'scale', 'C': 1}` | 0.6765 | 0.6736 | 0.5093 | 0.7554 |
| Sex_F | LogisticRegression | `{'solver': 'lbfgs', 'C': 0.01}` | 0.6841 | 0.6694 | 0.5556 | 0.7368 |
| Sex_F | DecisionTree | `{'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 30, 'criterion': 'entropy'}` | 0.6600 | 0.7521 | 0.6552 | 0.8065 |
| Sex_F | NeuralNetwork | `{'solver': 'adam', 'learning_rate_init': 0.001, 'hidden_layer_sizes': (100,), 'alpha': 0.001, 'activation': 'relu'}` | 0.6655 | 0.6529 | 0.4400 | 0.7485 |
