print("=== Export ICA 90-dimension Data with Named Columns ===")

import sys
sys.path.append('D:\\PythonLibs')

import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA

print("\n[1/3] Loading PCA data...")

# 加载PCA数据
pca_file = 'fMRI_PCA_499d_84p.csv'
df_pca = pd.read_csv(pca_file)
print(f"  PCA data: {df_pca.shape}")

# 保存患者ID
participant_ids = df_pca['participant_id']

# 提取特征（去掉ID列）
X = df_pca.drop(['participant_id'], axis=1).values
print(f"  Features: {X.shape[1]}")

print("\n[2/3] Applying ICA with 90 components...")

# ICA降维
ica = FastICA(n_components=90, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X)
print(f"  ICA output shape: {X_ica.shape}")

print("\n[3/3] Creating output file with named columns...")

# 创建列名：IC1, IC2, ..., IC90
ica_columns = [f'IC{i}' for i in range(1, 91)]

# 创建DataFrame，列名直接就是IC1-IC90
df_ica = pd.DataFrame(X_ica, columns=ica_columns)

# 插入患者ID作为第一列
df_ica.insert(0, 'participant_id', participant_ids.values)

# 保存为CSV
output_file = 'fMRI_ICA_90dim.csv'
df_ica.to_csv(output_file, index=False)

print(f"\n  File saved: {output_file}")
print(f"  Shape: {df_ica.shape[0]} rows, {df_ica.shape[1]} columns")
print("\n  Column names (The first row is the name of each dimension):")
print(f"  {list(df_ica.columns)}")

print("\n  Preview of first 5 rows:")
print(df_ica.head().to_string())

print("\n" + "="*60)
print("EXPORT COMPLETE")
print("="*60)
print(f"Output file: {output_file}")
print(f"File contains:")
print(f"  - Column 1: participant_id (participant ID)")
print(f"  - Columns 2-91: IC1 to IC90 (Names of 90 ICA dimensions)")
print(f"  - Total patients: {len(participant_ids)}")
print("\nPress Enter to exit...")
input()