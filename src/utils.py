import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv', header=None)
    print(data.head())
    idx = 1213
    matrix_data = data.iloc[idx, 1:].astype(float).values
    n = 200
    matrix = np.zeros((n, n))
    iu1 = np.triu_indices(n, k=1)
    matrix[iu1] = matrix_data
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 1)
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson Correlation')
    plt.title('fMRI Functional Connectome Matrix (200x200)')
    print(f"Restored matrix shape: {matrix.shape}")
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    plt.imshow(np.where(mask, matrix, np.nan), cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson Correlation (Upper Triangle)')
    plt.title('fMRI Functional Connectome Matrix Upper Triangle (200x200)')
    upper_triangle_data = matrix[iu1]
    upper_triangle_column = upper_triangle_data.reshape(-1, 1)
    plt.figure(figsize=(10, 4))
    plt.imshow(upper_triangle_column[0:100], cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('Flatten')
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
