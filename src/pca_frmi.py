# ============================================
# fMRI Data PCA with Visualization
# ============================================

print("Starting fMRI PCA with visualization...")

import sys
sys.path.append('D:\\PythonLibs')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("[1/6] Loading sample data...")
try:
    data_sample = pd.read_csv('data/processed/cleaned_train_connectome.csv', nrows=1000)
    print(f"  ✓ Loaded: {data_sample.shape[0]} samples, {data_sample.shape[1]} features")
except Exception as e:
    print(f"  ✗ Error: {e}")
    input("Press Enter to exit...")
    exit()

patient_ids = data_sample['participant_id']
features_sample = data_sample.drop('participant_id', axis=1)
original_features = features_sample.shape[1]

print("[2/6] Standardizing data...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_sample)

print("[3/6] Running full PCA analysis...")
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
marker_percents = [50, 75, 85, 90, 95]
colors = ['green', 'blue', 'orange', 'red', 'purple']

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
chart_file = 'PCA_Variance_Analysis_Chart.png'
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

# 自动选择85%
target_percent = 85
recommended_dim = dim_results.get(target_percent, 500)

print(f"\n[5/6] Using {recommended_dim} dimensions ({target_percent}% variance)...")

# 训练最终PCA
pca_final = PCA(n_components=recommended_dim)
pca_final.fit(features_scaled)

print("[6/6] Processing full dataset...")

# 分块处理
chunksize = 1000
all_chunks = []
total_processed = 0

try:
    for chunk_idx, chunk in enumerate(pd.read_csv('data/processed/cleaned_train_connectome.csv', 
                                                  chunksize=chunksize)):
        chunk_ids = chunk['participant_id']
        chunk_features = chunk.drop('participant_id', axis=1)
        
        chunk_scaled = scaler.transform(chunk_features)
        chunk_pca = pca_final.transform(chunk_scaled)
        
        chunk_df = pd.DataFrame(chunk_pca)
        chunk_df.columns = [f'PC{i+1}' for i in range(recommended_dim)]
        chunk_df.insert(0, 'participant_id', chunk_ids.values)
        
        all_chunks.append(chunk_df)
        total_processed += len(chunk_df)
        
        if (chunk_idx + 1) % 5 == 0:
            print(f"  Processed {total_processed} rows...")
            
except Exception as e:
    print(f"  ✗ Error during processing: {e}")
    input("Press Enter to exit...")
    exit()

# 合并保存
result_df = pd.concat(all_chunks, ignore_index=True)
variance_percent = np.sum(pca_final.explained_variance_ratio_) * 100
output_file = f'data/processed/fMRI_PCA_{recommended_dim}d_{int(variance_percent)}p.csv'
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