# Feature Ablation Study Results

本报告记录了在移除 `__missing_ind` 后、并严格对齐 XGBoost 最优参数的特征消融实验结果。

## 标签含义

- **ADHD_Outcome**
- Positive = 1 = ADHD
- Negative = 0 = Non-ADHD
- **Sex_F**
- Positive = 1 = Female
- Negative = 0 = Male

## 1. ADHD_Outcome

| Feature Combination | Acc | F1 (Positive=ADHD) | F1 (Negative=Non-ADHD) |
| :--- | :--- | :--- | :--- |
| All Features | 0.8058 | 0.8669 | 0.6412 |
| Cat + Quant | 0.8182 | 0.8750 | 0.6667 |
| Quant + fMRI | 0.7934 | 0.8571 | 0.6269 |
| Quantitative | 0.8058 | 0.8661 | 0.6466 |
| fMRI | 0.6818 | 0.8060 | 0.1149 |
| Categorical | 0.7025 | 0.8191 | 0.1628 |
| Cat + fMRI | 0.6983 | 0.8170 | 0.1412 |

## 2. Sex_F

| Feature Combination | Acc | F1 (Positive=Female) | F1 (Negative=Male) |
| :--- | :--- | :--- | :--- |
| All Features | 0.8058 | 0.6667 | 0.8630 |
| Cat + Quant | 0.7769 | 0.6494 | 0.8364 |
| Quantitative | 0.7851 | 0.6579 | 0.8434 |
| Quant + fMRI | 0.8140 | 0.6897 | 0.8673 |
| fMRI | 0.6612 | 0.3492 | 0.7709 |
| Cat + fMRI | 0.6777 | 0.3906 | 0.7809 |
| Categorical | 0.5744 | 0.3522 | 0.6831 |
