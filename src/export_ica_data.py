print("=== Export ICA 90-dimension Data with Named Columns ===")


import pandas as pd
import numpy as np
import os
from sklearn.decomposition import FastICA

print("\n[1/3] Loading PCA data...")

# 加载PCA数据
pca_file = 'data/reduced/fMRI_PCA_500d_combined.csv'
df_pca = pd.read_csv(pca_file)
print(f"  PCA data: {df_pca.shape}")

# 保存患者ID
participant_ids = df_pca['participant_id']

# 提取特征（去掉ID列）
X = df_pca.drop(['participant_id'], axis=1).values
print(f"  Features: {X.shape[1]}")

print("\n[2/3] Applying ICA with 100 components...")

# ICA降维
ica = FastICA(n_components=100, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X)
print(f"  ICA output shape: {X_ica.shape}")

print("\n[3/3] Creating output files and splitting datasets...")

# 创建列名：IC1, IC2, ..., IC100
ica_columns = [f'IC{i}' for i in range(1, 101)]

# 创建DataFrame
df_ica = pd.DataFrame(X_ica, columns=ica_columns)

# 插入患者ID作为第一列
df_ica.insert(0, 'participant_id', participant_ids.values)

# 确保输出目录存在
output_dir = 'data/reduced'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存完整合并后的ICA数据
full_output_file = os.path.join(output_dir, 'fMRI_ICA_100dim_combined.csv')
df_ica.to_csv(full_output_file, index=False)
print(f"  Full combined file saved: {full_output_file}")

# 拆分数据集 (训练集 1208, 测试集 304)
train_len = 1208
df_train_ica = df_ica.iloc[:train_len, :]
df_test_ica = df_ica.iloc[train_len:, :]

train_output_file = os.path.join(output_dir, 'train_fMRI_ICA_100dim.csv')
test_output_file = os.path.join(output_dir, 'test_fMRI_ICA_100dim.csv')

df_train_ica.to_csv(train_output_file, index=False)
df_test_ica.to_csv(test_output_file, index=False)

print(f"  Training set saved ({df_train_ica.shape[0]} rows): {train_output_file}")
print(f"  Test set saved ({df_test_ica.shape[0]} rows): {test_output_file}")

print(f"\n  Files saved in: {output_dir}")
print(f"  - Combined: {full_output_file}")
print(f"  - Training: {train_output_file}")
print(f"  - Test: {test_output_file}")
print("\n  Column names:")
print(f"  {list(df_ica.columns)}")

print("\n  Preview of first 5 rows (Combined):")
print(df_ica.head().to_string())

print("\n" + "="*60)
print("EXPORT AND SPLIT COMPLETE")
print("="*60)
print(f"Total patients processed: {len(participant_ids)}")
print(f"  - Training set: {train_len}")
print(f"  - Test set: {len(participant_ids) - train_len}")
print("\nPress Enter to exit...")
input()