# Single Label Independent RandomizedSearch Results

**Data shape:** X=(1208, 202), y_ADHD=(1208,), y_Sex=(1208,)

## Target Label: ADHD_Outcome

### Model: RandomForest

- **Best Parameters:** `{'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': None}`
- **Best CV F1 Score (Macro):** `0.7370`
- **Test Accuracy:** `0.7727`
- **Test F1 Score (Positive Class):** `0.8451`
- **Test F1 Score (Macro):** `0.7094`

### Model: XGBoost

- **Best Parameters:** `{'subsample': 1.0, 'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.2, 'colsample_bytree': 0.8}`
- **Best CV F1 Score (Macro):** `0.7500`
- **Test Accuracy:** `0.7934`
- **Test F1 Score (Positive Class):** `0.8563`
- **Test F1 Score (Macro):** `0.7443`

### Model: SVC

- **Best Parameters:** `{'kernel': 'rbf', 'gamma': 'scale', 'C': 1}`
- **Best CV F1 Score (Macro):** `0.6979`
- **Test Accuracy:** `0.7231`
- **Test F1 Score (Positive Class):** `0.7988`
- **Test F1 Score (Macro):** `0.6775`

### Model: LogisticRegression

- **Best Parameters:** `{'solver': 'lbfgs', 'C': 0.01}`
- **Best CV F1 Score (Macro):** `0.7224`
- **Test Accuracy:** `0.7438`
- **Test F1 Score (Positive Class):** `0.8063`
- **Test F1 Score (Macro):** `0.7141`

### Model: DecisionTree

- **Best Parameters:** `{'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 5, 'criterion': 'entropy'}`
- **Best CV F1 Score (Macro):** `0.6980`
- **Test Accuracy:** `0.7603`
- **Test F1 Score (Positive Class):** `0.8199`
- **Test F1 Score (Macro):** `0.7309`

### Model: NeuralNetwork

- **Best Parameters:** `{'solver': 'adam', 'learning_rate_init': 0.001, 'hidden_layer_sizes': (50, 50), 'alpha': 0.001, 'activation': 'relu'}`
- **Best CV F1 Score (Macro):** `0.6727`
- **Test Accuracy:** `0.7273`
- **Test F1 Score (Positive Class):** `0.8136`
- **Test F1 Score (Macro):** `0.6529`

---

## Target Label: Sex_F

### Model: RandomForest

- **Best Parameters:** `{'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 30}`
- **Best CV F1 Score (Macro):** `0.5652`
- **Test Accuracy:** `0.7066`
- **Test F1 Score (Positive Class):** `0.3238`
- **Test F1 Score (Macro):** `0.5682`

### Model: XGBoost

- **Best Parameters:** `{'subsample': 1.0, 'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.2, 'colsample_bytree': 0.8}`
- **Best CV F1 Score (Macro):** `0.7147`
- **Test Accuracy:** `0.7769`
- **Test F1 Score (Positive Class):** `0.6400`
- **Test F1 Score (Macro):** `0.7392`

### Model: SVC

- **Best Parameters:** `{'kernel': 'rbf', 'gamma': 'scale', 'C': 1}`
- **Best CV F1 Score (Macro):** `0.6677`
- **Test Accuracy:** `0.6860`
- **Test F1 Score (Positive Class):** `0.5422`
- **Test F1 Score (Macro):** `0.6516`

### Model: LogisticRegression

- **Best Parameters:** `{'solver': 'lbfgs', 'C': 0.01}`
- **Best CV F1 Score (Macro):** `0.6811`
- **Test Accuracy:** `0.6694`
- **Test F1 Score (Positive Class):** `0.5556`
- **Test F1 Score (Macro):** `0.6462`

### Model: DecisionTree

- **Best Parameters:** `{'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': 30, 'criterion': 'entropy'}`
- **Best CV F1 Score (Macro):** `0.6668`
- **Test Accuracy:** `0.7314`
- **Test F1 Score (Positive Class):** `0.6286`
- **Test F1 Score (Macro):** `0.7091`

### Model: NeuralNetwork

- **Best Parameters:** `{'solver': 'adam', 'learning_rate_init': 0.01, 'hidden_layer_sizes': (100,), 'alpha': 0.001, 'activation': 'relu'}`
- **Best CV F1 Score (Macro):** `0.6611`
- **Test Accuracy:** `0.7107`
- **Test F1 Score (Positive Class):** `0.4853`
- **Test F1 Score (Macro):** `0.6421`

---

