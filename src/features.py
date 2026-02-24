import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression


def pca_features(df: pd.DataFrame, n_components: int):
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(df)
    cols = [f"PC{i+1}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, columns=cols), pca


def select_k_best(X: pd.DataFrame, y: pd.Series, k: int):
    selector = SelectKBest(score_func=f_regression, k=k)
    Xt = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_cols = X.columns[mask]
    return pd.DataFrame(Xt, columns=selected_cols), selector
