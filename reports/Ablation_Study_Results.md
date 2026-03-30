# Feature Ablation Study Results

本报告记录了在移除“缺失值指示列 (`__missing_ind`)”且严格对齐网格搜索最佳超参数后的特征消融实验结果。模型采用 **XGBoost**。

## 1. ADHD_Outcome (ADHD 诊断预测)

| Feature Combination | Accuracy | F1 (Positive Class) | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| **All Features** | **0.8017** | **0.8636** | **0.7500** |
| Cat + Quant | 0.7893 | 0.8522 | 0.7426 |
| Quant + fMRI | 0.7727 | 0.8442 | 0.7122 |
| Quantitative | 0.7603 | 0.8304 | 0.7110 |
| fMRI | 0.6488 | 0.7709 | 0.5093 |
| Categorical | 0.6488 | 0.7632 | 0.5416 |
| Cat + fMRI | 0.6281 | 0.7541 | 0.4957 |

**核心发现**：
- **全部特征** (`All Features`) 达到了最高的准确率和 F1 分数。说明在剥离了缺失值噪声后，fMRI 脑电数据与基础量表数据形成了良好的互补。
- **量化特征** (`Quantitative`) 是预测 ADHD 的绝对核心。任何包含量化特征的组合，其 F1 分数均稳定在 0.83 以上。
- **分类特征** (`Categorical`) 单独使用时效果较差，且与 fMRI 结合 (`Cat + fMRI`) 时反而出现了性能下降，说明它们之间可能存在特征冲突或噪声叠加。

---

## 2. Sex_F (性别预测)

| Feature Combination | Accuracy | F1 (Positive Class) | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| **All Features** | **0.7893** | **0.6577** | **0.7527** |
| Quant + fMRI | 0.7645 | 0.6014 | 0.7171 |
| Quantitative | 0.7603 | 0.6133 | 0.7198 |
| Cat + Quant | 0.7562 | 0.6242 | 0.7219 |
| fMRI | 0.6983 | 0.4593 | 0.6250 |
| Cat + fMRI | 0.6612 | 0.4143 | 0.5880 |
| Categorical | 0.5579 | 0.3270 | 0.4989 |

**核心发现**：
- **全部特征** (`All Features`) 依然表现最好，特别是对少数类（女性）的识别能力 (`F1 Positive` 达到 0.6577)。
- **量化特征** (`Quantitative`) 在性别预测中同样起到了关键作用。
- **分类特征** (`Categorical`) 几乎无法用于预测性别（F1 仅为 0.3270），这符合逻辑，因为像“就诊年份”、“测试地点”等分类变量理论上与性别毫无因果关系。
