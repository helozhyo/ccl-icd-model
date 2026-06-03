#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

# 读取清理后的预测结果
pred_df = pd.read_csv('C:/Users/Hzh/Desktop/ccl-model-train/predictions_clean.csv')

# 生成提交文件：病案标识\t主要诊断编码|其他诊断编码|主要手术编码|其他手术编码
lines = []
for _, row in pred_df.iterrows():
    id_ = str(row['病案标识']).strip()
    main_d = str(row['主要诊断编码']).strip() if pd.notna(row['主要诊断编码']) else ''
    other_d = str(row['其他诊断编码']).strip() if pd.notna(row['其他诊断编码']) else ''
    main_s = str(row['主要手术编码']).strip() if pd.notna(row['主要手术编码']) else ''
    other_s = str(row['其他手术编码']).strip() if pd.notna(row['其他手术编码']) else ''

    # 字段内如果有残留|，只取第一部分
    def clean_field(s):
        if '|' in s:
            s = s.split('|')[0]
        return s.strip()

    main_d = clean_field(main_d)
    other_d = clean_field(other_d)
    main_s = clean_field(main_s)
    other_s = clean_field(other_s)

    line = f"{id_}\t{main_d}|{other_d}|{main_s}|{other_s}"
    lines.append(line)

# 写入txt
with open('C:/Users/Hzh/Desktop/ccl-model-train/submission.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f"已生成 submission.txt，共 {len(lines)} 条")
print("\n前15行预览:")
for line in lines[:15]:
    print(line)
print("\n后5行预览:")
for line in lines[-5:]:
    print(line)
