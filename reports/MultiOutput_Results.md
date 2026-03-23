# MultiOutput Classification Results

**Data shape:** X=(1208, 202), y=(1208, 2)

## RandomForest

- **Best Parameters:** `{'estimator__max_depth': 10, 'estimator__min_samples_split': 5, 'estimator__n_estimators': 100}`
- **Best CV F1 Score (Macro):** `0.5601`

### Label: ADHD_Outcome
- Accuracy: `0.7893`
- F1 Score (Positive Class): `0.8603`
- F1 Score (Macro): `0.7159`

### Label: Sex_F
- Accuracy: `0.6901`
- F1 Score (Positive Class): `0.2574`
- F1 Score (Macro): `0.5308`

**Overall Accuracy (Subset Accuracy):** `0.5455`

---

## XGBoost

- **Best Parameters:** `{'estimator__learning_rate': 0.1, 'estimator__max_depth': 5, 'estimator__n_estimators': 100}`
- **Best CV F1 Score (Macro):** `0.7209`

### Label: ADHD_Outcome
- Accuracy: `0.8140`
- F1 Score (Positive Class): `0.8725`
- F1 Score (Macro): `0.7645`

### Label: Sex_F
- Accuracy: `0.7769`
- F1 Score (Positive Class): `0.6400`
- F1 Score (Macro): `0.7392`

**Overall Accuracy (Subset Accuracy):** `0.6281`

---

## SVC

- **Best Parameters:** `{'estimator__C': 0.1, 'estimator__kernel': 'linear'}`
- **Best CV F1 Score (Macro):** `0.7054`

### Label: ADHD_Outcome
- Accuracy: `0.7521`
- F1 Score (Positive Class): `0.8125`
- F1 Score (Macro): `0.7233`

### Label: Sex_F
- Accuracy: `0.6694`
- F1 Score (Positive Class): `0.5556`
- F1 Score (Macro): `0.6462`

**Overall Accuracy (Subset Accuracy):** `0.5165`

---

