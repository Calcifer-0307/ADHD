import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
import os

def generate_ic_mapping_report():
    print("Loading Schaefer 200 labels...")
    schaefer_df = pd.read_csv("data/processed/Schaefer200_merged_labels(Schaefer200_merged_labels).csv")
    # 创建 ROI Label 到 Full Component Name 的映射字典
    schaefer_labels = dict(zip(schaefer_df['ROI Label'], schaefer_df['Full Component Name']))

    print("Loading raw functional connectome data...")
    # 1. 加载原始的脑网络连接矩阵数据
    fc_data = pd.read_csv('data/processed/cleaned_train_connectome.csv')
    
    # 获取原始的脑区连接特征名 (例如 0throw_1thcolumn)
    feature_names = [col for col in fc_data.columns if col != 'participant_id']
    X_fc = fc_data[feature_names].values
    
    print(f"Original shape: {X_fc.shape}")

    # 2. 重新拟合 PCA (模拟预处理时的步骤，保留解释 95% 方差的成分)
    print("Refitting PCA...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_fc)
    print(f"PCA components shape: {pca.components_.shape} (PCs x Original Features)")

    # 3. 重新拟合 ICA (提取 100 个独立成分)
    print("Refitting ICA...")
    ica = FastICA(n_components=100, random_state=42, max_iter=1000)
    X_ica = ica.fit_transform(X_pca)
    print(f"ICA components shape: {ica.components_.shape} (ICs x PCs)")
    
    # ========================================================
    # 4. 核心数学逻辑修正：计算复合权重 β (正向贡献)
    # 根据严格的推导：
    # IC = ica.components_ * PC
    # PC = pca.components_ * 原始连接
    # 所以: IC = (ica.components_ * pca.components_) * 原始连接
    # ========================================================
    print("Calculating composite weights (IC <- Original connections)...")
    
    # W (100 x n_PCs) dot V (n_PCs x n_Originals) = β (100 x n_Originals)
    composite_weights = np.dot(ica.components_, pca.components_)
    print(f"Composite weights shape: {composite_weights.shape}")
    
    output_md_path = 'data/processed/Detailed_IC_Network_Mapping.md'
    print(f"Generating detailed markdown report at {output_md_path}...")
    
    # 5. 写入 Markdown 文件
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("# Detailed Independent Component (IC) to Brain Network Mapping\n\n")
        f.write("This document lists the top 5 contributing original brain network connections for each Independent Component (IC).\n")
        f.write("Methodology: Composite weights = ica.components_ (W) × pca.components_ (V).\n\n")
        
        # composite_weights 的行数即为 IC 的数量 (100)
        for i in range(composite_weights.shape[0]):
            ic_name = f"IC{i+1}"
            weights = composite_weights[i]
            
            # 获取绝对值最大的 Top 5 权重的索引
            top_5_idx = np.argsort(np.abs(weights))[-5:][::-1]
            
            f.write(f"## {ic_name}\n")
            f.write("| Rank | Weight | Brain Region Connection |\n")
            f.write("|------|--------|-------------------------|\n")
            
            for rank, idx in enumerate(top_5_idx):
                weight_val = weights[idx]
                raw_name = feature_names[idx]
                
                # 解析 Xthrow_Ythcolumn 获取真实脑区名字
                region_name = raw_name
                if "throw_" in raw_name and "thcolumn" in raw_name:
                    try:
                        row_idx = int(raw_name.split("throw_")[0])
                        col_idx = int(raw_name.split("throw_")[1].replace("thcolumn", ""))
                        
                        # ROI Label 是 1-based, 索引需要 + 1
                        row_label = schaefer_labels.get(row_idx + 1, f"ROI_{row_idx+1}")
                        col_label = schaefer_labels.get(col_idx + 1, f"ROI_{col_idx+1}")
                        
                        region_name = f"{row_label} <--> {col_label}"
                    except Exception as e:
                        pass
                
                f.write(f"| {rank+1} | {weight_val:.6f} | {region_name} |\n")
            
            f.write("\n")
            
    print("Done! Report successfully generated.")

if __name__ == "__main__":
    generate_ic_mapping_report()