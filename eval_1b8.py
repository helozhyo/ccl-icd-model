#!/usr/bin/env python3
"""
评测预测结果，计算 M_total = 0.4*Acc_main + 0.1*F1_other_diag + 0.4*Acc_main_surg + 0.1*F1_other_surg
"""
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import json

PRED_FILE = "/root/autodl-tmp/predictions_1b8.csv"
LABEL_FILE = "/root/autodl-tmp/A_test.xlsx"  # 包含真实标签的Excel
OUTPUT_FILE = "/root/autodl-tmp/eval_result_1b8.json"


def safe_f1(y_true_list, y_pred_list, average='micro'):
    """计算F1，处理空列表情况"""
    # 移除空标签对
    valid = [(t, p) for t, p in zip(y_true_list, y_pred_list) if t or p]
    if not valid:
        return 0.0
    y_t = [set(x.split(';')) if x else set() for x, _ in valid]
    y_p = [set(x.split(';')) if x else set() for _, x in valid]

    # 宏F1：对每个集合内的元素计算
    all_labels = set()
    for s in y_t + y_p:
        all_labels.update(s)
    all_labels = sorted(all_labels)

    # 二元化
    y_t_bin = [[1 if lab in s else 0 for lab in all_labels] for s in y_t]
    y_p_bin = [[1 if lab in s else 0 for lab in all_labels] for s in y_p]

    if not all_labels:
        return 0.0
    return f1_score(y_t_bin, y_p_bin, average=average, zero_division=0)


def evaluate_predictions(pred_df, label_df):
    """计算评测指标"""
    # 合并预测与标签
    merged = pred_df.merge(label_df, on="病案标识", how="inner")
    print(f"合并后样本数: {len(merged)}")

    # 真实列名
    true_main_d = "主要诊断编码"
    true_other_d = "其他诊断编码"
    true_main_s = "主要手术编码"
    true_other_s = "其他手术编码"

    # 1. Acc_main (主要诊断精确匹配)
    acc_main = sum(
        1 for _, row in merged.iterrows()
        if str(row["主要诊断编码"]).strip() == str(row[true_main_d]).strip()
    ) / len(merged) if len(merged) > 0 else 0

    # 2. F1_other_diag (其他诊断F1)
    y_true_other_d = [str(row[true_other_d]) if pd.notna(row[true_other_d]) else "" for _, row in merged.iterrows()]
    y_pred_other_d = [str(row["其他诊断编码"]) if pd.notna(row["其他诊断编码"]) else "" for _, row in merged.iterrows()]
    f1_other_d = safe_f1(y_true_other_d, y_pred_other_d)

    # 3. Acc_main_surg (主要手术精确匹配)
    acc_main_surg = sum(
        1 for _, row in merged.iterrows()
        if str(row["主要手术编码"]).strip() == str(row[true_main_s]).strip()
    ) / len(merged) if len(merged) > 0 else 0

    # 4. F1_other_surg (其他手术F1，最多2个)
    y_true_other_s = [str(row[true_other_s]) if pd.notna(row[true_other_s]) else "" for _, row in merged.iterrows()]
    y_pred_other_s = [str(row["其他手术编码"]) if pd.notna(row["其他手术编码"]) else "" for _, row in merged.iterrows()]
    f1_other_s = safe_f1(y_true_other_s, y_pred_other_s)

    # M_total
    M_total = 0.4 * acc_main + 0.1 * f1_other_d + 0.4 * acc_main_surg + 0.1 * f1_other_s

    results = {
        "样本数": len(merged),
        "Acc_main": acc_main,
        "F1_other_diag": f1_other_d,
        "Acc_main_surg": acc_main_surg,
        "F1_other_surg": f1_other_s,
        "M_total": M_total,
    }

    return results


def main():
    print("加载预测结果...")
    pred_df = pd.read_csv(PRED_FILE)
    print(f"预测样本数: {len(pred_df)}")

    print("加载真实标签...")
    label_df = pd.read_excel(LABEL_FILE, engine='openpyxl')
    print(f"标签样本数: {len(label_df)}")
    print(f"标签列: {label_df.columns.tolist()}")

    # 统一列名
    if "病案标识" in label_df.columns:
        label_df = label_df.rename(columns={label_df.columns[0]: "病案标识"})

    results = evaluate_predictions(pred_df, label_df)

    print("\n" + "=" * 50)
    print("评测结果:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
