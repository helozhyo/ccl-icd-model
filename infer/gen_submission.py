#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import re

# 读取预测结果
pred_df = pd.read_csv('C:/Users/Hzh/Desktop/ccl-model-train/predictions_1b8.csv')

# 清理函数
def clean_code(s):
    if pd.isna(s):
        return ''
    s = str(s).strip()
    # 移除 <|im_end|> 等残留
    s = re.sub(r'<\|[^|]+\|>', '', s)
    # 移除行尾的 <
    s = re.sub(r'<$', '', s)
    s = s.strip('|')
    # 移除空白
    s = re.sub(r'\s+', '', s)
    # 如果为空返回空字符串
    if not s or s in ['无', '空', '']:
        return ''
    return s


# 生成提交文件
lines = []
for _, row in pred_df.iterrows():
    id_ = str(row['病案标识']).strip()
    main_d = clean_code(row['主要诊断编码'])
    other_d = clean_code(row['其他诊断编码'])
    main_s = clean_code(row['主要手术编码'])
    other_s = clean_code(row['其他手术编码'])

    # 格式: 病案标识\t主要诊断编码|其他诊断编码|主要手术编码|其他手术编码
    line = f"{id_}\t{main_d}|{other_d}|{main_s}|{other_s}"
    lines.append(line)

# 写入txt
with open('C:/Users/Hzh/Desktop/ccl-model-train/submission.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f"已生成 submission.txt，共 {len(lines)} 条")
print("\n前10行预览:")
for line in lines[:10]:
    print(line)
