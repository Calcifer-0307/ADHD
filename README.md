# ADHD Female Brain Project

本项目用于探究**性别因素对 ADHD（注意力缺陷多动障碍）诊断的潜在影响**。通过结合基础人口学特征、临床量表特征与高维 fMRI（功能性磁共振成像）脑网络特征，构建多模态机器学习模型，旨在揭示现有诊断指标在不同性别群体中的表现差异与偏见。

## 目录结构

```text
ADHD_Female_Brain_Project/
├── data/                    # 数据目录（不参与 Git 同步）
│   ├── raw/                 # 原始 Excel/CSV 数据 (需手动放入)
│   └── processed/           # 清洗、插补、编码及降维后的特征文件 (自动生成)
├── notebooks/               # 探索性实验 Jupyter Notebooks
├── reports/                 # 实验报告和高质量混淆矩阵图像 (自动生成)
│   ├── Ablation_Study_Results.md                  # 特征消融实验报告
│   ├── SingleLabel_RandomizedSearch_Results.md    # 模型网格搜索最佳参数与性能报告
│   ├── SingleLabel_RandomizedSearch_Summary_Table.md # 单标签结果汇总表
│   └── Confusion_Matrix/                          # 按目标标签分类的高级混淆矩阵图像
├── output/                  # 临时输出目录 (存放 PCA 可视化图表等)
├── scripts/                 # 命令行训练入口脚本
│   └── train_model.py       # 基于 CSV 输入的训练脚本
├── src/                     # 核心项目代码（模块化）
│   ├── __init__.py
│   ├── baseline_models.py   # 基线多输出模型封装
│   ├── get_data.py          # 数据加载与特征拼接模块 (保证全流程数据对齐)
│   ├── preprocessing.py     # 数据预处理脚本 (数据清洗、MICE多重插补、One-Hot编码)
│   ├── model_train.py       # 模型训练主脚本 (单标签独立网格搜索、随机森林/XGBoost等6种模型对比)
│   ├── ablation_study.py    # 特征消融实验 (量化Categorical, Quantitative, fMRI三大模块的贡献)
│   ├── advanced_cm.py       # 高级混淆矩阵可视化工具 (含精确率、召回率、FDR、FNR等边缘统计)
│   ├── pca_fmri.py          # fMRI PCA 降维脚本 (支持合并训练/测试集统一降维)
│   ├── ica_analysis.py      # fMRI ICA 分析脚本
│   └── export_ica_data.py   # 导出 ICA 特征
├── requirements.txt         # 项目依赖库列表
└── README.md                # 项目说明文档
```

## 环境准备 (Environment Setup)

本项目支持在 macOS 和 Windows 系统上运行。为了避免依赖冲突，建议使用 Python 虚拟环境。

### 1. 创建虚拟环境

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**注意 (macOS 用户):**
如果遇到 `xgboost` 相关的 `libomp` 错误，请先使用 Homebrew 安装 OpenMP 运行时库：
```bash
brew install libomp
```

## 数据准备 (Data Preparation)

请将原始数据文件放入 `data/raw/` 目录中。所需核心文件包括：
- `TRAIN_QUANTITATIVE_METADATA_new.xlsx`
- `TRAIN_CATEGORICAL_METADATA_new.xlsx`
- `TRAINING_SOLUTIONS.xlsx`
- `TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv`

## 核心运行流程 (Execution Pipeline)

### 1. 数据预处理 (Preprocessing)

运行预处理脚本，进行缺失值过滤、MICE 多重插补、分类变量独热编码，并生成干净的特征文件至 `data/processed/`。
*(注：最新版本已移除对模型产生噪声的 `__missing_ind` 缺失值指示列)*

```bash
python src/preprocessing.py --input_dir data/raw --outdir data/processed
```

### 2. fMRI 特征提取 (PCA + ICA)

对高维的 fMRI 功能连接矩阵进行分步降维。为了保证特征空间的统一性，流程如下：

1.  **合并与 PCA**: 运行 `pca_fmri.py`，将训练集 (1208 样本) 与测试集 (304 样本) 合并为 1512*19900 的矩阵，进行标准化后执行 PCA 降维至 **500 维**。生成的方差分析图表将保存至 `output/`。
    ```bash
    python src/pca_fmri.py
    ```
2.  **ICA 进一步降维**: 运行 `export_ica_data.py`，对 PCA 后的 500 维特征进行独立成分分析 (ICA)，进一步压缩至 **100 维**。
3.  **拆分数据集**: 降维完成后，脚本会自动将 1512 个样本拆回原始的训练集和测试集比例，并保存为最终的特征文件。
    ```bash
    python src/export_ica_data.py
    ```

### 3. 模型训练与评估 (Training & Evaluation)

本项目包含多种机器学习模型（XGBoost, Random Forest, SVC, Logistic Regression, Decision Tree, Neural Network），并结合高级混淆矩阵与消融实验，从多角度评估不同模态特征对两个目标标签的贡献。

**主训练管线 (网格搜索与模型对比):**
该脚本分别针对 `ADHD_Outcome` 和 `Sex_F` 执行随机网格搜索，寻找各自的最优参数，自动生成高级混淆矩阵，并汇总结果至 `reports/SingleLabel_RandomizedSearch_Results.md`。
```bash
python src/model_train.py
```

**结果汇总表:**
为了便于横向比较不同模型在两个标签上的表现，项目还提供单表汇总文件：
`reports/SingleLabel_RandomizedSearch_Summary_Table.md`

**特征消融实验 (Feature Ablation Study):**
基于 `model_train.py` 中 XGBoost 的最优参数，按 `ADHD_Outcome` 与 `Sex_F` 两个 label 分别加载各自最佳参数，并测试“分类特征”、“量化特征”、“fMRI特征”及其不同组合对预测性能的贡献度，结果输出至 `reports/Ablation_Study_Results.md`。
```bash
python src/ablation_study.py
```

**基线训练脚本:**
若需要通过命令行直接读取特征 CSV 与目标 CSV 进行训练，可使用：
```bash
python scripts/train_model.py --features <features.csv> --targets <targets.csv>
```

## 常见问题 (Troubleshooting)

- **数据对齐问题**: 在进行自定义特征切片时，请务必统一使用 `src/get_data.py` 中的逻辑获取 `X` 和 `y`，避免使用 `pd.merge` 的内连接导致样本无声丢失，从而破坏验证集的分布一致性。
- **Windows 编码问题**: 如果遇到 `UnicodeDecodeError`，请尝试在打开文件时指定 `encoding='utf-8'`。
- **评价结果解读**: 请勿仅参考 `Accuracy`（准确率），在不平衡数据下，高准确率可能伴随较高的漏诊率。建议结合分类报告、汇总表与高级混淆矩阵共同判断模型表现。
