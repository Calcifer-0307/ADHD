import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def read_xlsx(path, sheet): # 读取Excel文件的指定工作表，返回一个DataFrame
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

def clamp_df(df, low, high): # 将DataFrame中的数值列限制在指定的范围内，返回一个新的DataFrame和修改的单元格数量。
    clamped = df.clip(lower=low, upper=high)
    diff = ((df < low) | (df > high)).sum().sum()
    return clamped, int(diff)

def strip_obj(df):  # 去除字符串前后的空格，并将变为空字符串的单元格转换为缺失值（NaN），同时统计总共修改了多少个单元格。
    changed = 0
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            before = s.copy()
            s = s.str.strip()
            changed += int((before != s).sum())
            df[c] = s.replace({"": np.nan})
    return df, changed

def fill_categorical(df): # 填充DataFrame中的分类变量缺失值，返回一个新的DataFrame和审计信息。
    audit = {"categorical": {"missing": {}, "unknown_added": {}, "strip_changed": 0, "indicators": 0}}
    df2, changed = strip_obj(df.copy())
    audit["categorical"]["strip_changed"] = int(changed)
    tokens = {"", "na", "n/a", "null", "none", "nan"}
    for c in df2.columns:
        if c == "participant_id":
            continue
        if not pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].apply(lambda x: np.nan if isinstance(x, str) and x.strip().lower() in tokens else x)
            m = df2[c].isna().sum()
            audit["categorical"]["missing"][c] = int(m)
            ind = f"{c}__missing_ind"
            df2[ind] = df2[c].isna().astype(int)
            audit["categorical"]["indicators"] += int(df2[ind].sum())
            df2[c] = df2[c].fillna("Unknown")
            audit["categorical"]["unknown_added"][c] = int(m)
        else:
            m = df2[c].isna().sum()
            audit["categorical"]["missing"][c] = int(m)
            ind = f"{c}__missing_ind"
            df2[ind] = df2[c].isna().astype(int)
            audit["categorical"]["indicators"] += int(df2[ind].sum())
            if m > 0:
                mode = df2[c].mode(dropna=True)
                fillv = mode.iloc[0] if len(mode) > 0 else -1
                df2[c] = df2[c].fillna(fillv)
    return df2, audit

def clean_connectome(df): # 清理DataFrame中的连接矩阵数据，返回一个新的DataFrame和审计信息。
    clamped_df, changed = clamp_df(df.drop(columns=["participant_id"]), -1.0, 1.0)
    if clamped_df.isna().sum().sum() > 0:
        clamped_df = clamped_df.fillna(0.0)
    cleaned = pd.concat([df[["participant_id"]], clamped_df], axis=1)
    audit = {"connectome": {"clamped_values": int(changed), "nan_filled": int((clamped_df.isna().sum().sum()))}}
    return cleaned, audit

def clean_solutions(df): # 清理DataFrame中的解决方案数据，返回一个新的DataFrame和审计信息。
    """
    Clean the solutions dataframe:
    - Strip whitespace from participant_id
    - Remove duplicates
    - Ensure numeric types for outcome columns
    """
    audit = {"solutions": {"duplicates_dropped": 0, "strip_changed": 0}}
    
    if "participant_id" in df.columns:
        # Strip whitespace
        s = df["participant_id"].astype(str)
        before = s.copy()
        s = s.str.strip()
        audit["solutions"]["strip_changed"] = int((before != s).sum())
        df["participant_id"] = s
        
        # Drop duplicates
        before_len = len(df)
        df = df.drop_duplicates(subset=["participant_id"], keep="first")
        audit["solutions"]["duplicates_dropped"] = before_len - len(df)
        
    return df, audit

def _norm_missing_tokens(df): # 归一化DataFrame中的缺失值标记，将其转换为NaN，返回一个新的DataFrame。
    tokens = {"", "na", "n/a", "null", "none", "nan"}
    df2 = df.copy()
    for c in df2.columns:
        if c == "participant_id":
            continue
        if not pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].apply(lambda x: np.nan if isinstance(x, str) and x.strip().lower() in tokens else x)
    return df2

def _id_missing_ratio(df): # 计算DataFrame中每个患者的缺失值比例，返回一个Series。
    if "participant_id" not in df.columns:
        return pd.Series(dtype=float)
    df2 = _norm_missing_tokens(df)
    cols = [c for c in df2.columns if c != "participant_id"]
    if not cols:
        return pd.Series(dtype=float)
    miss = df2[cols].isna().sum(axis=1)
    total = len(cols)
    ratios = miss / float(total)
    return pd.Series(ratios.values, index=df2["participant_id"])

def filter_ids_by_missing(train_q, train_cat, train_fc, train_sol, threshold): # 根据缺失值比例过滤患者ID，返回过滤后的DataFrame和审计信息。
    r_q = _id_missing_ratio(train_q)
    r_cat = _id_missing_ratio(train_cat)
    ids = set(train_q["participant_id"]).union(set(train_cat["participant_id"]))
    avg = {}
    for pid in ids:
        vals = []
        if pid in r_q.index:
            vals.append(float(r_q.loc[pid]))
        if pid in r_cat.index:
            vals.append(float(r_cat.loc[pid]))
        if vals:
            avg[pid] = sum(vals) / len(vals)
    remove = {pid for pid, v in avg.items() if v > threshold}
    audit = {"threshold": threshold, "removed_count": len(remove), "removed_ids": sorted(list(remove))}
    if len(remove) > 0:
        train_q = train_q[~train_q["participant_id"].isin(remove)]
        train_cat = train_cat[~train_cat["participant_id"].isin(remove)]
        train_sol = train_sol[~train_sol["participant_id"].isin(remove)]
        train_fc = train_fc[~train_fc["participant_id"].isin(remove)]
    return train_q, train_cat, train_fc, train_sol, audit

def _select_mice_columns(df, min_ratio=0.05, max_ratio=0.4): # 选择MICE填充的列，根据缺失值比例返回目标列和预测列。
    numeric_cols = [c for c in df.columns if c != "participant_id" and pd.api.types.is_numeric_dtype(df[c])]
    ratios = {c: float(df[c].isna().mean()) for c in numeric_cols}
    selected = [c for c, r in ratios.items() if r >= min_ratio and r <= max_ratio]
    predictors = [c for c, r in ratios.items() if r < max_ratio]
    return selected, predictors

def fit_apply_mice(train_df, target_cols, predictor_cols): # 使用MICE填充训练集的目标列，返回填充后的DataFrame和审计信息。
    used_cols = sorted(set(target_cols).union(set(predictor_cols)))
    train_mat = train_df[used_cols].copy()
    imp = IterativeImputer(sample_posterior=True, random_state=42)
    imp.fit(train_mat)
    imputed_train = imp.transform(train_mat)
    train_imputed_df = pd.DataFrame(imputed_train, columns=used_cols, index=train_df.index)
    for c in target_cols:
        train_df[c] = train_imputed_df[c]
    return train_df, {"mice_cols": target_cols, "predictors": predictor_cols}

def fill_quantitative(train_q, train_cat, group_cols, skip_fill_cols=None): # 填充训练集的定量数据，根据分组列和全局中位数填充缺失值，返回填充后的DataFrame和审计信息。
    if skip_fill_cols is None:
        skip_fill_cols = set()
    else:
        skip_fill_cols = set(skip_fill_cols)
    audit = {"quantitative": {"missing": {}, "strategy": {}, "inserted_indicators": 0}}
    merged = train_q.merge(train_cat[group_cols + ["participant_id"]], on="participant_id", how="left")
    group_stats = {}
    global_medians = {}
    for c in train_q.columns:
        if c == "participant_id":
            continue
        if pd.api.types.is_numeric_dtype(train_q[c]):
            m = train_q[c].isna().sum()
            audit["quantitative"]["missing"][c] = int(m)
            indicator = f"{c}__missing_ind"
            merged[indicator] = merged[c].isna().astype(int)
            audit["quantitative"]["inserted_indicators"] += int(merged[indicator].sum())
            gm = train_q[c].median()
            global_medians[c] = float(gm) if not np.isnan(gm) else None
            grp = merged.groupby(group_cols)[c].median() if group_cols else None
            if grp is not None:
                group_stats[c] = grp.to_dict()
            if c in skip_fill_cols:
                audit["quantitative"]["strategy"][c] = {"group": group_cols, "group_median_used": 0, "global_median": None, "skipped_for_mice": True}
                continue
            if grp is not None:
                merged[c] = merged.apply(lambda r: grp.get(tuple(r[g] for g in group_cols), np.nan) if pd.isna(r[c]) else r[c], axis=1)
            merged[c] = merged[c].fillna(gm)
            audit["quantitative"]["strategy"][c] = {"group": group_cols, "group_median_used": int(m), "global_median": float(gm) if not np.isnan(gm) else None}
    base_cols = [c for c in train_q.columns if c != "participant_id"]
    ind_cols = [c for c in merged.columns if c.endswith("__missing_ind")]
    cols = ["participant_id"] + base_cols + ind_cols
    return merged[cols], audit, group_stats, global_medians

def finalize_no_nan_quant(df, global_medians): # 填充DataFrame中的缺失值，根据全局中位数填充，返回填充后的DataFrame和审计信息。
    filled = {}
    for c in df.columns:
        if c == "participant_id" or c.endswith("__missing_ind"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            before = int(df[c].isna().sum())
            if before > 0:
                fillv = global_medians.get(c, None)
                if fillv is None or np.isnan(fillv):
                    fillv = 0
                df[c] = df[c].fillna(fillv)
            after = int(df[c].isna().sum())
            filled[c] = before - after
    return df, {"final_quant_filled": filled}

def finalize_no_nan_cat(df): # 填充DataFrame中的缺失值，根据全局中位数填充，返回填充后的DataFrame和审计信息。
    filled = {}
    df2 = df.copy()
    for c in df2.columns:
        if c == "participant_id" or c.endswith("__missing_ind"):
            continue
        if pd.api.types.is_numeric_dtype(df2[c]):
            before = int(df2[c].isna().sum())
            if before > 0:
                mode = df2[c].mode(dropna=True)
                fillv = mode.iloc[0] if len(mode) > 0 else -1
                df2[c] = df2[c].fillna(fillv)
            after = int(df2[c].isna().sum())
            filled[c] = before - after
        else:
            before = int(df2[c].isna().sum())
            if before > 0:
                df2[c] = df2[c].fillna("Unknown")
            after = int(df2[c].isna().sum())
            filled[c] = before - after
    return df2, {"final_cat_filled": filled}

def one_hot_encode(train_cat_df): # 对DataFrame中的分类变量进行独热编码，返回编码后的DataFrame和审计信息。
    """
    Perform one-hot encoding on categorical columns.
    Ensures that numeric columns intended as categorical (e.g., site ID) are treated as strings before encoding.
    Excludes specific columns (e.g., Sex_F, Basic_Demos_Enroll_Year) from being one-hot encoded.
    """
    # Define columns to completely exclude from One-Hot Encoding
    exclude_from_ohe = ["Sex_F", "Basic_Demos_Enroll_Year"]
    
    # Identify base columns (excluding ID, missing indicators, and excluded columns)
    train_cols = [c for c in train_cat_df.columns if c != "participant_id"]
    base_cols = [c for c in train_cols if not c.endswith("__missing_ind") and c not in exclude_from_ohe]
    ind_cols = [c for c in train_cols if c.endswith("__missing_ind")]
    
    # Explicitly cast base columns to string to ensure get_dummies treats them as categorical
    encoded_df = pd.get_dummies(train_cat_df[base_cols].astype(str), dtype=int)
    
    # Get the excluded columns that actually exist in the dataframe to keep them intact
    kept_cols = [c for c in exclude_from_ohe if c in train_cat_df.columns]
    
    # Concatenate: participant_id + kept original columns + one-hot features + missing indicators
    comb_out = pd.concat([train_cat_df[["participant_id"] + kept_cols], encoded_df, train_cat_df[ind_cols]], axis=1)
    
    return comb_out, {"columns": base_cols, "indicator_columns": ind_cols, "features": list(encoded_df.columns)}

def preprocess(root=".", outdir="cleaned_data", use_mice=True, mice_min_ratio=0.05, mice_max_ratio=0.4, input_dir="data/raw"):
    root = os.path.abspath(root)
    input_path = os.path.join(root, input_dir)
    os.makedirs(os.path.join(root, outdir), exist_ok=True)
    
    # Check if input_dir exists, if not fallback to TRAIN_NEW in root (legacy)
    train_q_path = os.path.join(input_path, "TRAIN_QUANTITATIVE_METADATA_new.xlsx")
    if not os.path.exists(train_q_path):
        input_path = os.path.join(root, "TRAIN_NEW")

    train_q = read_xlsx(os.path.join(input_path, "TRAIN_QUANTITATIVE_METADATA_new.xlsx"), "training_combined")
    train_cat = read_xlsx(os.path.join(input_path, "TRAIN_CATEGORICAL_METADATA_new.xlsx"), "training_combined")
    train_sol = read_xlsx(os.path.join(input_path, "TRAINING_SOLUTIONS.xlsx"), "training_combined")
    train_fc = pd.read_csv(os.path.join(input_path, "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"))
    
    # Clean solutions
    train_sol, audit_sol = clean_solutions(train_sol)
    
    train_cat = train_cat.merge(train_sol[["participant_id", "Sex_F"]], on="participant_id", how="left")
    train_q, train_cat, train_fc, train_sol, audit_remove_train = filter_ids_by_missing(train_q, train_cat, train_fc, train_sol, 0.3)
    
    group_cols = []
    if "Basic_Demos_Study_Site" in train_cat.columns:
        group_cols.append("Basic_Demos_Study_Site")
    if "Sex_F" in train_cat.columns:
        group_cols.append("Sex_F")
    mice_audit = {}
    mice_cols = []
    pred_cols = []
    if use_mice:
        mice_cols, pred_cols = _select_mice_columns(train_q, min_ratio=mice_min_ratio, max_ratio=mice_max_ratio)
        train_q, _, mice_info = fit_apply_mice(train_q, pd.DataFrame(columns=train_q.columns), mice_cols, pred_cols)
        mice_audit = {"selected_cols": mice_info["mice_cols"], "predictors": mice_info["predictors"], "min_ratio": mice_min_ratio, "max_ratio": mice_max_ratio}
    cleaned_train_q, audit_train_q, group_stats, global_medians = fill_quantitative(train_q, train_cat, group_cols, skip_fill_cols=mice_cols)
    cleaned_train_q, final_q_train = finalize_no_nan_quant(cleaned_train_q, global_medians)
    
    cleaned_train_cat, audit_train_cat = fill_categorical(train_cat)
    cleaned_train_cat, final_cat_train = finalize_no_nan_cat(cleaned_train_cat)
    
    cleaned_train_fc, audit_train_fc = clean_connectome(train_fc)
    
    tr_ohe, ohe_info = one_hot_encode(cleaned_train_cat)
    
    cleaned_train_q.to_csv(os.path.join(root, outdir, "cleaned_train_quantitative.csv"), index=False)
    # cleaned_train_cat.to_csv(os.path.join(root, outdir, "cleaned_train_categorical.csv"), index=False)
    cleaned_train_fc.to_csv(os.path.join(root, outdir, "cleaned_train_connectome.csv"), index=False)
    tr_ohe.to_csv(os.path.join(root, outdir, "cleaned_train_categorical_ohe.csv"), index=False)
    train_sol.to_csv(os.path.join(root, outdir, "cleaned_train_solutions.csv"), index=False)
    
    with open(os.path.join(root, outdir, "ohe_categories.json"), "w", encoding="utf-8") as fh:
        json.dump(ohe_info, fh, ensure_ascii=False, indent=2)
        
    audit = {"remove_train": audit_remove_train, "mice": mice_audit, "train_quantitative": audit_train_q, "train_categorical": audit_train_cat, "train_connectome": audit_train_fc, "final_quant_train": final_q_train, "final_cat_train": final_cat_train, "solutions_cleaning": audit_sol}
    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, "reports", "cleaning_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)
    return {"train_q": cleaned_train_q, "train_cat": cleaned_train_cat, "train_fc": cleaned_train_fc, "train_ohe": tr_ohe, "ohe_info": ohe_info, "audit": audit}

def run_preprocessing(input_dir="data/raw", output_dir="data/processed", categorical_cols=None): # 运行预处理管线，读取原始数据，进行清洗和转换，输出处理后的数据和审计信息。
    """
    对齐脚本入口的接口：
    - 当前清洗管线读取 data/raw 目录中的原始文件 (TRAIN_NEW compatible)
    - 输出目录统一为 data/processed，返回绝对路径字符串
    """
    preprocess(root=".", outdir=output_dir, use_mice=True, mice_min_ratio=0.05, mice_max_ratio=0.4, input_dir=input_dir)
    return os.path.join(os.path.abspath("."), output_dir)

def main(): # 主函数，解析命令行参数，调用预处理管线。
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="cleaned_data")
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--use_mice", action="store_true")
    parser.add_argument("--mice_min_ratio", type=float, default=0.05)
    parser.add_argument("--mice_max_ratio", type=float, default=0.4)
    args = parser.parse_args()
    preprocess(root=args.root, outdir=args.outdir, use_mice=args.use_mice, mice_min_ratio=args.mice_min_ratio, mice_max_ratio=args.mice_max_ratio, input_dir=args.input_dir)
    print(os.path.join(os.path.abspath(args.root), args.outdir))

if __name__ == "__main__":
    main()
