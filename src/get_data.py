import pandas as pd
import numpy as np

def get_data():
    # 读取categorical data文件，读取为一个DataFrame
    categorical_df = pd.read_csv("data/processed/cleaned_train_categorical_ohe.csv")

    # 读取quantitative data文件，读取为一个DataFrame
    quantitative_df = pd.read_csv("data/processed/cleaned_train_quantitative.csv")

    # 读取降维后的connectome文件，读取为一个DataFrame
    connectome_df = pd.read_csv("data/processed/fMRI_ICA_100dim.csv")

    # 读取train_solution文件，读取为一个DataFrame
    labels_df = pd.read_csv("data/processed/cleaned_train_solutions.csv")

    # 提取patient id至一个列表
    patient_ids = labels_df["participant_id"].tolist()

    # 遍历patient id列表，读取每个患者的categorical data、quantitative data、connectome data，合并为一个特征向量
    X = []
    y1 = [] # ADHD_Outcome
    y2 = [] # Sex_F
    drop_cols = ["participant_id", "Sex_F", "Basic_Demos_Enroll_Year"]
    for pid in patient_ids:
        try:
            cat_data = categorical_df[categorical_df["participant_id"] == pid].drop(columns=drop_cols).values.flatten()
            quant_data = quantitative_df[quantitative_df["participant_id"] == pid].drop(columns=["participant_id"]).values.flatten()
            conn_data = connectome_df[connectome_df["participant_id"] == pid].drop(columns=["participant_id"]).values.flatten()

            combined_features = np.concatenate([cat_data, quant_data, conn_data])
            
            label1_val = labels_df[labels_df["participant_id"] == pid]["ADHD_Outcome"].values[0]
            label2_val = labels_df[labels_df["participant_id"] == pid]["Sex_F"].values[0]
            X.append(combined_features)
            y1.append(label1_val)
            y2.append(label2_val)
        except IndexError:
            continue
            
    X = np.array(X)
    y1 = np.array(y1)
    y2 = np.array(y2)

    return X, y1, y2
