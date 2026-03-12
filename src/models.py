import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from get_data import get_data





def train_and_evaluate_models(X, y):
    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 构造SVM模型
    svc_model = SVC(kernel="rbf", random_state=42)
    rf_model  = RandomForestClassifier()
    xgboost_model = XGBClassifier()

    # 训练模型
    svc_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgboost_model.fit(X_train, y_train)

    # 预测测试集
    y_pred_svc = svc_model.predict(X_test)
    y_pred_rf  = rf_model.predict(X_test)
    y_pred_xgboost = xgboost_model.predict(X_test)

    # 计算准确率
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    accuracy_rf  = accuracy_score(y_test, y_pred_rf)
    accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)

    # 输出分类报告
    print("SVC Classification Report:")
    print(classification_report(y_test, y_pred_svc))
    print("RF Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgboost))
    
    return {
        "svc": svc_model,
        "rf": rf_model,
        "xgboost": xgboost_model,
        "metrics": {
            "svc_acc": accuracy_svc,
            "rf_acc": accuracy_rf,
            "xgboost_acc": accuracy_xgboost
        }
    }

if __name__ == "__main__":
   X, y = get_data()
   train_and_evaluate_models(X, y)
