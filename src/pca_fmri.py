# ============================================
# fMRI Data PCA with Visualization
# ============================================

print("Starting fMRI PCA with visualization...")

import sys
# sys.path.append('D:\\PythonLibs')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("[1/6] Loading data...")
try:
    # 加载训练集 (1208 samples)
    train_data = pd.read_csv('data/processed/cleaned_train_connectome.csv')
    print(f"  ✓ Loaded training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    
    # 加载测试集 (304 samples)
    test_data = pd.read_csv('data/raw/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
    print(f"  ✓ Loaded test data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
    
    # 合并数据集 (1208 + 304 = 1512)
    combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    print(f"  ✓ Combined data: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")
    
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    exit()

patient_ids = combined_data['participant_id']
features_combined = combined_data.drop('participant_id', axis=1)
original_features = features_combined.shape[1]

print("[2/6] Standardizing data...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_combined)

print("[3/6] Running full PCA analysis...")
# 因为样本数较少 (1512)，我们可以直接对全量数据进行 PCA
pca_full = PCA()
pca_full.fit(features_scaled)

explained_ratio = pca_full.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_ratio)

print("\n" + "="*70)
print("PCA VARIANCE ANALYSIS RESULTS")
print("="*70)

# 分析每个百分比
targets = [50, 60, 70, 75, 80, 85, 90, 95]
dim_results = {}

print("\nDimensions needed for each variance level:")
print("-" * 55)

for target_percent in targets:
    target = target_percent / 100.0
    for i, ratio in enumerate(cumulative_ratio, 1):
        if ratio >= target:
            dim_results[target_percent] = i
            
            # 简单文本进度条
            progress = int(30 * target_percent / 100)
            bar = "[" + "■" * progress + "□" * (30 - progress) + "]"
            
            print(f"  {target_percent:3d}% variance: {i:4d} dimensions {bar}")
            break

print("\n[4/6] Creating visualization charts...")

# 创建图表
plt.figure(figsize=(14, 5))

# 图表1：累计解释率曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_ratio)+1), cumulative_ratio * 100, 
         'b-', linewidth=2, alpha=0.7)
plt.xlabel('Number of Principal Components', fontsize=11)
plt.ylabel('Cumulative Variance Explained (%)', fontsize=11)
plt.title('PCA: How Many Dimensions for How Much Variance?', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 标记关键点
marker_percents = [50, 75, 80, 85, 90, 95]
colors = ['green', 'blue', 'cyan', 'orange', 'red', 'purple']

for percent, color in zip(marker_percents, colors):
    if percent in dim_results:
        dim = dim_results[percent]
        plt.plot(dim, percent, 'o', color=color, markersize=10, 
                label=f'{percent}%: {dim} dim')
        plt.axvline(x=dim, color=color, linestyle='--', alpha=0.3)
        plt.axhline(y=percent, color=color, linestyle='--', alpha=0.3)

plt.legend(loc='lower right')
plt.xlim(0, 1000)

# 图表2：维度需求柱状图
plt.subplot(1, 2, 2)
percentages = list(dim_results.keys())
dimensions = list(dim_results.values())

bars = plt.bar(range(len(percentages)), dimensions, 
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(percentages))))
plt.xticks(range(len(percentages)), [f'{p}%' for p in percentages])
plt.xlabel('Variance Target (%)', fontsize=11)
plt.ylabel('Dimensions Required', fontsize=11)
plt.title('Dimensions Needed for Different Variance Levels', 
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 在柱子上显示数字
for bar, dim in zip(bars, dimensions):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 20, 
             str(dim), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# 保存图表
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"  ✓ Created directory: {output_dir}")

chart_file = os.path.join(output_dir, 'PCA_Variance_Analysis_Chart.png')
plt.savefig(chart_file, dpi=150, bbox_inches='tight')
print(f"  ✓ Chart saved as: {chart_file}")

# 显示推荐
print("\n" + "="*70)
print("RECOMMENDATION FOR YOUR PROJECT")
print("="*70)

print(f"\nOriginal features: {original_features:,}")
print("\nSuggested options:")
print("  1. 499 dimensions → 85% variance (balanced, recommended)")
print("  2. 615 dimensions → 90% variance (higher quality)")
print("  3. 406 dimensions → 80% variance (more efficient)")

# 自动选择 500 维 (或 85% 变异度推荐的维度)
target_percent = 85
recommended_dim = 500 # 强制指定为 500 维，如用户要求

print(f"\n[5/6] Using {recommended_dim} dimensions...")

# 训练最终PCA
pca_final = PCA(n_components=recommended_dim)
pca_transformed = pca_final.fit_transform(features_scaled)

print("[6/6] Saving combined PCA results...")

# 创建 DataFrame
result_df = pd.DataFrame(pca_transformed)
result_df.columns = [f'PC{i+1}' for i in range(recommended_dim)]
result_df.insert(0, 'participant_id', patient_ids.values)

# 合并保存
variance_percent = np.sum(pca_final.explained_variance_ratio_) * 100
output_file = f'data/processed/fMRI_PCA_{recommended_dim}d_combined.csv'
result_df.to_csv(output_file, index=False)

print("\n" + "="*70)
print(" PCA DIMENSIONALITY REDUCTION COMPLETE!")
print("="*70)
print(f"Original data: {original_features:,} features")
print(f"Reduced to: {recommended_dim} principal components")
print(f"Variance explained: {variance_percent:.2f}%")
print(f"Output file: {output_file}")
print(f"Estimated size: {recommended_dim * 1209 * 8 / 1024 / 1024:.1f} MB")
print(f"Chart saved: {chart_file}")
print("\nFiles created:")
print(f"  1. {chart_file} - Visualization chart")
print(f"  2. {output_file} - PCA-reduced data for model training")
print("\nYour PCA task is complete! ")
print("\nPress Enter to exit...")
input()