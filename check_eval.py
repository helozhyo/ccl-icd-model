#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import json

# Check A_test labels
df = pd.read_excel('/root/autodl-tmp/A_test.xlsx', engine='openpyxl')
print("Columns:", df.columns.tolist())
print()
print("First 3 rows, first 5 cols:")
for i in range(3):
    row = df.iloc[i]
    print(f"  {row.iloc[0]}: {row.iloc[1]} | {row.iloc[2]} | {row.iloc[3]} | {row.iloc[4]}")

print()
# Check predictions
pred_df = pd.read_csv('C:/Users/Hzh/Desktop/ccl-model-train/predictions_1b8.csv')
print("Pred columns:", pred_df.columns.tolist())
print("First 3 rows:")
for i in range(3):
    row = pred_df.iloc[i]
    print(f"  {row['病案标识']}: {row['主要诊断编码']} | {row['其他诊断编码']} | {row['主要手术编码']} | {row['其他手术编码']}")

print()
# Manual comparison for first 5
print("=== First 5 comparison ===")
for i in range(5):
    label_id = str(df.iloc[i, 0])
    pred_row = pred_df[pred_df['病案标识'] == label_id]
    if len(pred_row) == 0:
        pred_row = pred_df[pred_df['病案标识'].astype(str) == label_id]
    if len(pred_row) == 0:
        pred_row = pred_df.iloc[i]
    else:
        pred_row = pred_row.iloc[0]

    label_main_d = str(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else ""
    label_other_d = str(df.iloc[i, 2]) if pd.notna(df.iloc[i, 2]) else ""
    label_main_s = str(df.iloc[i, 3]) if pd.notna(df.iloc[i, 3]) else ""
    label_other_s = str(df.iloc[i, 4]) if pd.notna(df.iloc[i, 4]) else ""

    pred_main_d = str(pred_row['主要诊断编码']) if pd.notna(pred_row['主要诊断编码']) else ""
    pred_other_d = str(pred_row['其他诊断编码']) if pd.notna(pred_row['其他诊断编码']) else ""
    pred_main_s = str(pred_row['主要手术编码']) if pd.notna(pred_row['主要手术编码']) else ""
    pred_other_s = str(pred_row['其他手术编码']) if pd.notna(pred_row['其他手术编码']) else ""

    print(f"\n[{i}] ID={label_id}")
    print(f"  Label main_d: {repr(label_main_d[:50])}")
    print(f"  Pred  main_d: {repr(pred_main_d[:50])}")
    print(f"  Match: {label_main_d.strip() == pred_main_d.strip()}")
    print(f"  Label other_d: {repr(label_other_d[:60])}")
    print(f"  Pred  other_d: {repr(pred_other_d[:60])}")
    print(f"  Match: {set(label_other_d.split(';')) == set(pred_other_d.split(';')) if label_other_d and pred_other_d else label_other_d == pred_other_s}")
