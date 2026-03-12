# ADHD Female Brain Project

本项目用于 ADHD 女性脑数据的探索、预处理、特征工程与多输出模型训练。

## 目录结构

```
ADHD_Female_Brain_Project/
├── data/                    # 数据目录（不参与 Git 同步）
│   ├── raw/                 # 原始 Excel/CSV (需手动放入)
│   └── processed/           # 清洗后的文件与 PCA 特征 (自动生成)
├── notebooks/               # 探索性实验
│   └── 01_EDA_Visualization.ipynb
├── src/                     # 项目代码（模块化）
│   ├── __init__.py
│   ├── get_data.py          # 数据加载模块
│   ├── preprocessing.py     # 数据预处理脚本 (清洗、编码、缺失值处理)
│   ├── features.py          # PCA 与特征选择
│   ├── models.py            # 模型定义与训练逻辑
│   ├── train.py             # 主训练脚本
│   ├── pca_frmi.py          # fMRI PCA 降维脚本
│   ├── ica_analysis.py      # fMRI ICA 分析脚本
│   └── utils.py             # 辅助工具
├── .gitignore
├── requirements.txt
└── README.md
```

## 环境准备 (Environment Setup)

### 1. 创建虚拟环境

为了避免依赖冲突，建议使用 Python 虚拟环境。

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
如果遇到 `xgboost` 相关的 `libomp` 错误，请先安装 OpenMP 运行时库：

```bash
brew install libomp
```

## 数据准备 (Data Preparation)

请将原始数据文件放入 `data/raw/` 目录中。所需文件包括：

- `TRAIN_QUANTITATIVE_METADATA_new.xlsx`
- `TRAIN_CATEGORICAL_METADATA_new.xlsx`
- `TRAINING_SOLUTIONS.xlsx`
- `TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv`

## 运行流程 (Execution)

### 1. 数据预处理 (Preprocessing)

运行预处理脚本，清洗数据并生成处理后的文件至 `data/processed/`。

```bash
python src/preprocessing.py --input_dir data/raw --outdir data/processed
```

### 2. 特征降维 ( PCA  --> ICA)

对 fMRI 数据进行降维处理：

**PCA 分析:**

```bash
python src/pca_frmi.py
```

**ICA 对比 PCA 分析:**

```bash
python src/ica_analysis.py
```

**导出 ICA 数据 (用于模型训练):**

```bash
python src/export_ica_data.py
```

*注意：在进行模型训练前，如果选择使用 ICA 降维后的数据，请务必运行此脚本生成特征文件。*

### 3. 模型训练与评估 (Training)

运行主训练脚本，加载处理后的数据并训练模型（SVM, Random Forest, XGBoost）。

```bash
python src/models.py
```

或者

```bash
python src/train.py
```

## 常见问题 (Troubleshooting)

- **Windows 编码问题**: 如果遇到 `UnicodeDecodeError`，请尝试在打开文件时指定 `encoding='utf-8'`。
- **路径分隔符**: 本项目代码使用 `os.path.join` 处理路径，兼容 Windows (`\`) 和 macOS/Linux (`/`)。
- **XGBoost 报错**: Windows 用户通常不需要额外安装库，但 macOS 用户必须安装 `libomp`。

