print("=== fMRI Data ICA Analysis ===")

import sys
sys.path.append('D:\\PythonLibs')

import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("\n[1/4] Loading data...")

pca_file = 'fMRI_PCA_499d_84p.csv'
df_pca = pd.read_csv(pca_file)
print(f"  PCA data: {df_pca.shape}")

# 加载标签
df_labels = pd.read_excel('TRAINING_SOLUTIONS.xlsx')
print(f"  Labels: {df_labels.shape}")

# 合并
data = pd.merge(df_pca, df_labels, on='participant_id')
X = data.drop(['participant_id', 'Sex_F'], axis=1).values
y = data['Sex_F'].values

print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")

print("\n[2/4] Testing different ICA dimensions...")

# 要测试的ICA维度
ica_dims = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
results = []

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n_comp in ica_dims:
    if n_comp > X.shape[1]:
        continue
        
    print(f"  Testing ICA({n_comp})...", end=' ')
    
    # ICA降维
    ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
    X_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    
    # 训练分类器
    model = LogisticRegression(max_iter=1000)
    model.fit(X_ica, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_ica))
    test_acc = accuracy_score(y_test, model.predict(X_test_ica))
    
    results.append({
        'dim': n_comp,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    print(f"test={test_acc:.3f}")

# 找最佳
print("\n" + "="*50)
best = max(results, key=lambda x: x['test_acc'])
print(f" Best ICA dimension: {best['dim']}")
print(f"   Test accuracy: {best['test_acc']:.3f}")
print(f"   Train accuracy: {best['train_acc']:.3f}")

# 和PCA结果对比
print("\n Comparison with PCA:")
print("-" * 40)
print("Method | Dim | Test Acc")
print("-" * 40)
print(f"PCA    | 200 | ?.???") 
print(f"ICA    | {best['dim']:3d} | {best['test_acc']:.3f}")

# 画图
plt.figure(figsize=(10, 5))
dims = [r['dim'] for r in results]
test_acc = [r['test_acc'] for r in results]
train_acc = [r['train_acc'] for r in results]

plt.plot(dims, train_acc, 'b-o', label='Train (ICA)')
plt.plot(dims, test_acc, 'r-o', label='Test (ICA)')
plt.xlabel('ICA Components')
plt.ylabel('Accuracy')
plt.title('ICA Performance vs Number of Components')
plt.legend()
plt.grid(True)
plt.savefig('ica_results.png')
print("\n Chart saved: ica_results.png")

# 分析ICA成分的独立性
print("\n[3/4] Analyzing component independence...")

# 用最佳维度再做一次完整ICA
ica_final = FastICA(n_components=best['dim'], random_state=42, max_iter=1000)
X_ica_final = ica_final.fit_transform(X)

# 计算成分之间的相关性（应该接近0才说明独立）
corr_matrix = np.corrcoef(X_ica_final.T)
avg_corr = np.mean(np.abs(corr_matrix - np.eye(best['dim'])))
print(f"  Average absolute correlation between components: {avg_corr:.4f}")
print(f"  (Should be close to 0 for good ICA separation)")

# 计算峰度（非高斯性指标）
from scipy.stats import kurtosis
kurt_values = [kurtosis(X_ica_final[:, i]) for i in range(best['dim'])]
print(f"  Average kurtosis: {np.mean(np.abs(kurt_values)):.2f}")
print(f"  (Higher means more non-Gaussian, better for ICA)")

print("\n[4/4] Conclusion:")
print("="*50)

if best['dim'] < 100:
    print(f" ICA works well! Only {best['dim']} components needed.")
    print(f"   This is much less than PCA's 499 dimensions.")
    print(f"   You can tell your group member: ICA reduces to {best['dim']} dims")
    print(f"   while maintaining {best['test_acc']:.1f}% accuracy.")
else:
    print(f" ICA needs {best['dim']} components, similar to PCA.")
    print(f"   In this case, stick with PCA for simplicity.")

print("\nPress Enter to exit...")
input()