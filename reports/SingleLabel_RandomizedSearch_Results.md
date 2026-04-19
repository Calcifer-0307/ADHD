# Single Label Independent RandomizedSearch Results

**Data shape:** X=(1208, 174), y_ADHD=(1208,), y_Sex=(1208,)

## Target Label: ADHD_Outcome

### Model: RandomForest

- **Best Parameters:** `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 10}`
- **Best CV F1 Score (Macro):** `0.7320`
- **Test Accuracy:** `0.7810`
- **Test F1 Score (Positive Class):** `0.8499`
- **Test F1 Score (Negative Class):** `0.5954`

### Model: XGBoost

- **Best Parameters:** `{'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 1.0}`
- **Best CV F1 Score (Macro):** `0.7466`
- **Test Accuracy:** `0.8058`
- **Test F1 Score (Positive Class):** `0.8669`
- **Test F1 Score (Negative Class):** `0.6412`

### Model: SVC

- **Best Parameters:** `{'kernel': 'rbf', 'gamma': 'scale', 'C': 1}`
- **Best CV F1 Score (Macro):** `0.7046`
- **Test Accuracy:** `0.7149`
- **Test F1 Score (Positive Class):** `0.7928`
- **Test F1 Score (Negative Class):** `0.5430`

### Model: LogisticRegression

- **Best Parameters:** `{'solver': 'lbfgs', 'C': 0.01}`
- **Best CV F1 Score (Macro):** `0.7219`
- **Test Accuracy:** `0.7479`
- **Test F1 Score (Positive Class):** `0.8076`
- **Test F1 Score (Negative Class):** `0.6347`

### Model: DecisionTree

- **Best Parameters:** `{'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 5, 'criterion': 'entropy'}`
- **Best CV F1 Score (Macro):** `0.6957`
- **Test Accuracy:** `0.7603`
- **Test F1 Score (Positive Class):** `0.8199`
- **Test F1 Score (Negative Class):** `0.6420`

### Model: NeuralNetwork

- **Best Parameters:** `{'solver': 'adam', 'learning_rate_init': 0.01, 'hidden_layer_sizes': (100,), 'alpha': 0.001, 'activation': 'relu'}`
- **Best CV F1 Score (Macro):** `0.6805`
- **Test Accuracy:** `0.7438`
- **Test F1 Score (Positive Class):** `0.8218`
- **Test F1 Score (Negative Class):** `0.5441`

---

## Target Label: Sex_F

### Model: RandomForest

- **Best Parameters:** `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10}`
- **Best CV F1 Score (Macro):** `0.5653`
- **Test Accuracy:** `0.6942`
- **Test F1 Score (Positive Class):** `0.3148`
- **Test F1 Score (Negative Class):** `0.8032`

### Model: XGBoost

- **Best Parameters:** `{'subsample': 1.0, 'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.8}`
- **Best CV F1 Score (Macro):** `0.7069`
- **Test Accuracy:** `0.8058`
- **Test F1 Score (Positive Class):** `0.6667`
- **Test F1 Score (Negative Class):** `0.8630`

### Model: SVC

- **Best Parameters:** `{'kernel': 'rbf', 'gamma': 'scale', 'C': 1}`
- **Best CV F1 Score (Macro):** `0.6765`
- **Test Accuracy:** `0.6736`
- **Test F1 Score (Positive Class):** `0.5093`
- **Test F1 Score (Negative Class):** `0.7554`

### Model: LogisticRegression

- **Best Parameters:** `{'solver': 'lbfgs', 'C': 0.01}`
- **Best CV F1 Score (Macro):** `0.6841`
- **Test Accuracy:** `0.6694`
- **Test F1 Score (Positive Class):** `0.5556`
- **Test F1 Score (Negative Class):** `0.7368`

### Model: DecisionTree

- **Best Parameters:** `{'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 30, 'criterion': 'entropy'}`
- **Best CV F1 Score (Macro):** `0.6600`
- **Test Accuracy:** `0.7521`
- **Test F1 Score (Positive Class):** `0.6552`
- **Test F1 Score (Negative Class):** `0.8065`

### Model: NeuralNetwork

- **Best Parameters:** `{'solver': 'adam', 'learning_rate_init': 0.001, 'hidden_layer_sizes': (100,), 'alpha': 0.001, 'activation': 'relu'}`
- **Best CV F1 Score (Macro):** `0.6655`
- **Test Accuracy:** `0.6529`
- **Test F1 Score (Positive Class):** `0.4400`
- **Test F1 Score (Negative Class):** `0.7485`

---

