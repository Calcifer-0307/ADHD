import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from get_data import get_data

# 1. 严格使用原始 get_data 函数，保证样本顺序和数量与网格搜索时 100% 一致
# 这里的 X 已经是不包含 Sex_F 和 Year 的纯特征矩阵了
X, y_adhd, y_sex = get_data(keep_sex=False)

# 2. 动态计算三大特征块的索引范围 (避免硬编码)
# 我们只需要知道各表去除了哪些列，就能算出它们的宽度
cat_df = pd.read_csv("data/processed/cleaned_train_categorical_ohe.csv")
quant_df = pd.read_csv("data/processed/cleaned_train_quantitative.csv")
fmri_df = pd.read_csv("data/processed/fMRI_ICA_100dim.csv")

# cat_df 在 get_data(keep_sex=False) 中去掉了 "participant_id", "Sex_F", "Basic_Demos_Enroll_Year"
n_cat = cat_df.shape[1] - 3 
n_quant = quant_df.shape[1] - 1
n_fmri = fmri_df.shape[1] - 1

# 定义切片边界
cat_idx = (0, n_cat)
quant_idx = (n_cat, n_cat + n_quant)
fmri_idx = (n_cat + n_quant, n_cat + n_quant + n_fmri)

# 3. 定义组合 (保存需要保留的特征列索引)
combinations = {
    "Categorical": list(range(*cat_idx)),
    "Quantitative": list(range(*quant_idx)),
    "fMRI": list(range(*fmri_idx)),
    "Cat + Quant": list(range(*cat_idx)) + list(range(*quant_idx)),
    "Cat + fMRI": list(range(*cat_idx)) + list(range(*fmri_idx)),
    "Quant + fMRI": list(range(*quant_idx)) + list(range(*fmri_idx)),
    "All Features": list(range(X.shape[1])) # 所有列
}

def run_ablation(target_name, y_target):
    print("="*70)
    print(f"特征消融实验 (XGBoost 预测 {target_name})")
    print(f"{'特征组合':<20} | {'Acc':<10} | {'F1(Positive)':<15} | {'F1(Macro)':<10}")
    print("-" * 70)

    for name, cols in combinations.items():
        # 根据组合切片提取特定的特征列
        X_sub = X[:, cols]
        
        # 必须加入 StandardScaler，与网格搜索保持一致
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_target, test_size=0.2, random_state=42 # 移除 stratify，与 model_train.py 保持完全一致的随机切分
        )
        
        # 控制变量保持模型参数一致
        clf = XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            subsample=1.0,
            n_estimators=500,
            max_depth=5,
            learning_rate=0.2,
            colsample_bytree=0.8
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_pos = f1_score(y_test, y_pred, zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"{name:<20} | {acc:<10.4f} | {f1_pos:<15.4f} | {f1_mac:<10.4f}")
    print("="*70 + "\n")

# 4. 分别对 ADHD 和 Sex_F 运行消融实验
run_ablation("ADHD_Outcome", y_adhd)
run_ablation("Sex_F (Gender)", y_sex)
