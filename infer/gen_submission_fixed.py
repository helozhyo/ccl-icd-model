#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成提交文件，使用更合理的ICD编码清理逻辑
"""
import pandas as pd
import re

pred_df = pd.read_csv('C:/Users/Hzh/Desktop/ccl-model-train/predictions_1b8.csv')

# ---------- 工具函数 ----------

def is_icd10_diag(code):
    """ICD-10 诊断：字母开头，如 A00, C50.900x011, Z98.800x612, I10.x00"""
    return bool(re.match(r'^[A-Z][0-9]', code))

def is_icd9_surg(code):
    """ICD-9 手术：数字开头，如 85.4301, 99.2503, 06.3100x002"""
    return bool(re.match(r'^[0-9][0-9]', code))

def is_valid_code(code):
    """判断是否像有效的ICD编码"""
    return is_icd10_diag(code) or is_icd9_surg(code)

def clean_single_code(s):
    """清理单个编码：去除残留字符，只保留字母数字点号和x"""
    s = str(s).strip()
    # 去除行尾的 < 等残留
    s = re.sub(r'[<\[,，].*$', '', s)
    # 去除 <|im_end|> 等特殊token残留
    s = re.sub(r'\|?im_end[|>]*', '', s)
    s = re.sub(r'<\|[^|]+\|>', '', s)
    s = re.sub(r'\|?endoftext', '', s)
    # 去除行尾空白
    s = s.strip().rstrip('<').strip()
    return s

def clean_codes_field(s, is_surgery=False):
    """
    清理一整个编码字段（如多个诊断分号分隔，或多个手术分号分隔）
    只保留有效的ICD编码，去掉中文描述和乱码
    """
    if pd.isna(s) or str(s).strip() == '':
        return ''

    s = str(s).strip()
    # 先处理 | 分隔的情况，只取第一部分
    if '|' in s:
        s = s.split('|')[0]

    # 去掉 < 之后的全部内容
    s = re.sub(r'<.*$', '', s)

    # 分割各编码（用分号或逗号分隔）
    raw_codes = re.split(r'[;,，]', s)
    valid = []
    for c in raw_codes:
        c = clean_single_code(c)
        if not c:
            continue
        # 跳过纯中文（超过30%中文字符）
        chinese_chars = sum(1 for ch in c if '一' <= ch <= '鿿')
        if chinese_chars > len(c) * 0.3:
            continue
        # 跳过纯英文描述（没有数字的）
        if re.match(r'^[A-Za-z\s]+$', c):
            continue
        # 必须是有效的ICD编码
        if is_valid_code(c):
            valid.append(c)

    return ';'.join(valid)


# ---------- 生成提交文件 ----------
lines = []
for _, row in pred_df.iterrows():
    id_ = str(row['病案标识']).strip()

    # 四个字段分别清理
    # strip 之后还要去除 \r（CSV读取时可能带入）
    main_d = clean_codes_field(row['主要诊断编码'], is_surgery=False).replace('\r', '').strip()
    other_d = clean_codes_field(row['其他诊断编码'], is_surgery=False).replace('\r', '').strip()
    main_s = clean_codes_field(row['主要手术编码'], is_surgery=True).replace('\r', '').strip()
    other_s = clean_codes_field(row['其他手术编码'], is_surgery=True).replace('\r', '').strip()
    line = f"{id_}\t{main_d}|{other_d}|{main_s}|{other_s}"
    lines.append(line)

# 写入
with open('C:/Users/Hzh/Desktop/ccl-model-train/submission.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

# ---------- 验证 ----------
print(f"已生成 submission.txt，共 {len(lines)} 条")
print("\n前15行预览:")
for line in lines[:15]:
    print(line)
print("\n后5行预览:")
for line in lines[-5:]:
    print(line)

# 统计主诊断分布
import pandas as pd
main_ds = [l.split('\t')[1].split('|')[0] for l in lines]
from collections import Counter
cnt = Counter(main_ds)
print(f"\n主诊断唯一值数: {len(cnt)}")
print("主诊断分布(前10):")
for code, n in cnt.most_common(10):
    print(f"  {code}: {n}")
