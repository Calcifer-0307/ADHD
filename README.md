# ADHD Female Brain Project

本项目用于 ADHD 女性脑数据的探索、预处理、特征工程与多输出模型训练。

## 目录结构

```
ADHD_Female_Brain_Project/
├── data/                    # 数据目录（不参与 Git 同步）
│   ├── raw/                 # 原始 Excel/CSV
│   └── processed/           # 清洗后的文件与 PCA 特征
├── notebooks/               # 探索性实验
│   └── 01_EDA_Visualization.ipynb
├── src/                     # 项目代码（模块化）
│   ├── __init__.py
│   ├── data_loader.py       # 接口：通过 patient_id 获取特征
│   ├── preprocessing.py     # 清洗、编码、缺失值处理
│   ├── features.py          # PCA 与特征选择
│   ├── models.py            # 多输出模型定义与训练
│   └── utils.py             # 可视化辅助方法
├── scripts/                 # 可执行脚本
│   ├── run_preprocessing.py # 运行完整清洗管线
│   └── train_model.py       # 训练与评估
├── .gitignore
├── requirements.txt
└── README.md
```

## 快速开始

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 准备数据：
   - 将原始数据放入 `data/raw/`
3. 运行预处理：
   ```
   python scripts/run_preprocessing.py --input_dir data/raw --output_dir data/processed
   ```
4. 训练模型（示例）：
   ```
   python scripts/train_model.py --features data/processed/features.csv --targets data/processed/targets.csv
   ```

说明：具体数据列命名与目标文件需根据实际数据调整。上述脚手架提供统一入口与模块化代码结构，便于协作开发与复现。
