import os
import argparse
import numpy as np
import pandas as pd
from .preprocessing import preprocess

class PatientVectorLoader:
    def __init__(self, data_dir="data/processed"):
        """
        统一数据目录至 data/processed，读取清洗后的训练集产物：
        - cleaned_train_quantitative.csv
        - cleaned_train_categorical_ohe.csv
        - cleaned_train_connectome.csv
        """
        self.data_dir = data_dir
        self.q = None
        self.c = None
        self.f = None
        self.q_cols = None
        self.c_cols = None
        self.f_cols = None

    def load(self):
        q = pd.read_csv(os.path.join(self.data_dir, "cleaned_train_quantitative.csv")).drop_duplicates(subset=["participant_id"], keep="first").set_index("participant_id")
        c = pd.read_csv(os.path.join(self.data_dir, "cleaned_train_categorical_ohe.csv")).drop_duplicates(subset=["participant_id"], keep="first").set_index("participant_id")
        f = pd.read_csv(os.path.join(self.data_dir, "cleaned_train_connectome.csv")).drop_duplicates(subset=["participant_id"], keep="first").set_index("participant_id")
        self.q_cols = [col for col in q.columns if not col.endswith("__missing_ind")]
        self.c_cols = list(c.columns)
        self.f_cols = list(f.columns)
        self.q = q
        self.c = c
        self.f = f

    def get_by_id(self, participant_id):
        if self.q is None or self.c is None or self.f is None:
            self.load()
        rq = self.q.loc[participant_id] if participant_id in self.q.index else None
        rc = self.c.loc[participant_id] if participant_id in self.c.index else None
        rf = self.f.loc[participant_id] if participant_id in self.f.index else None
        if rq is None and rc is None and rf is None:
            return None, None, None
        q_vec = rq[self.q_cols].to_numpy().astype(float).flatten() if rq is not None else np.array([])
        c_vec = rc[self.c_cols].to_numpy().astype(float).flatten() if rc is not None else np.array([])
        f_vec = rf[self.f_cols].to_numpy().astype(float).flatten() if rf is not None else np.array([])
        return q_vec, c_vec, f_vec

_LOADER = None
def get_data_by_id(participant_id):
    global _LOADER
    if _LOADER is None:
        _LOADER = PatientVectorLoader("data/processed")
        _LOADER.load()
    return _LOADER.get_by_id(participant_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--use_mice", action="store_true")
    parser.add_argument("--mice_min_ratio", type=float, default=0.05)
    parser.add_argument("--mice_max_ratio", type=float, default=0.4)
    parser.add_argument("--patient_id", default=None)
    args = parser.parse_args()
    if args.preprocess:
        preprocess(root=args.root, outdir=args.data_dir, use_mice=args.use_mice, mice_min_ratio=args.mice_min_ratio, mice_max_ratio=args.mice_max_ratio)
    loader = PatientVectorLoader(args.data_dir)
    loader.load()
    pid = args.patient_id
    if pid:
        q, c, f = loader.get_by_id(pid)
        if q is None and c is None and f is None:
            print('{"error": "participant_id not found"}')
        else:
            print(f"Quantitative Vector: {q.shape}")
            print(f"Categorical Vector: {c.shape}")
            print(f"Connectome Matrix: {f.shape}")
        return
    while True:
        try:
            pid = input().strip()
        except EOFError:
            break
        if not pid or pid.lower() in ("q", "quit", "exit"):
            break
        q, c, f = loader.get_by_id(pid)
        if q is None and c is None and f is None:
            print('{"error": "participant_id not found"}')
            continue
        print(f"Quantitative Vector: {q.shape}")
        print(f"Categorical Vector: {c.shape}")
        print(f"Connectome Matrix: {f.shape}")

if __name__ == "__main__":
    main()
