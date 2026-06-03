#!/usr/bin/env python3
"""
CCL 2026 ICD自动编码任务 - 数据预处理脚本
将Excel数据转换为模型训练格式
"""

import os
import sys
import json
import pandas as pd
import openpyxl

TRAIN_FILE = "/root/autodl-tmp/train.xlsx"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
OUTPUT_DIR = "/root/autodl-tmp/icd_data"

# 列名映射（按位置索引）
# 0=病案标识, 1=主诉, 2=现病史, 3=既往史, 4=个人史, 5=婚姻史, 6=家族史,
# 7=入院情况, 8=入院诊断, 9=诊疗经过, 10=出院情况, 11=出院医嘱,
# 12=手术经过, 13=术前诊断, 14=术中诊断,
# 15=主要诊断编码, 16=其他诊断编码, 17=主要手术编码, 18=其他手术编码

TEXT_COLS = list(range(1, 15))  # 1-14列: 所有文本字段
LABEL_COLS = {
    'main_diag': 15,      # 主要诊断编码
    'other_diag': 16,     # 其他诊断编码
    'main_surg': 17,      # 主要手术编码
    'other_surg': 18,     # 其他手术编码
}

# 预定义的编码类别
MAIN_DIAG_CLASSES = [
    'C50.900x011', 'I20.000', 'I48.x02', 'I63.900', 'I66.901',
    'J18.900', 'J98.414', 'K31.703', 'K92.901', 'M80.900',
    'N18.900x013', 'N40.x00', 'O34.201', 'R91.x02', 'S32.000x002',
    'C73.x00', 'Z51.100', 'Z51.102'
]

MAIN_SURG_CLASSES = [
    '74.1x01', '88.4101', '99.2503', '06.3100x002', '43.4105',
    '32.2400x002', '55.6901', '85.4301', '37.3401', '45.1300x004',
    '45.1600x001', '81.6600x001', '00.6600x008', '60.2901',
    '60.2100x001', '33.2403'
]

def read_excel(path, sheet=1):
    """读取Excel，跳过标题行"""
    xl = pd.ExcelFile(path, engine='openpyxl')
    # Sheet1 或第2个sheet
    name = 'Sheet1' if 'Sheet1' in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(name)
    return df

def concat_text(row):
    """将所有文本字段拼接成一个字符串"""
    parts = []
    for col_idx in TEXT_COLS:
        val = row.iloc[col_idx]
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return '\n'.join(parts)

def format_output(main_diag, other_diag, main_surg, other_surg):
    """
    按要求格式输出: 主要诊断编码|其他诊断编码1;其他诊断编码2;…|主要手术诊断编码|其他手术编码1;其他手术编码2;…
    """
    other_d_str = other_diag if pd.notna(other_diag) and str(other_diag).strip() else ''
    other_s_str = other_surg if pd.notna(other_surg) and str(other_surg).strip() else ''

    return f"{main_diag}|{other_d_str}|{main_surg}|{other_s_str}"

def process_data(df, is_train=True):
    """处理数据，返回[{text, labels}]列表"""
    records = []
    for idx, row in df.iterrows():
        text = concat_text(row)

        if is_train:
            main_diag = row.iloc[LABEL_COLS['main_diag']]
            other_diag = row.iloc[LABEL_COLS['other_diag']]
            main_surg = row.iloc[LABEL_COLS['main_surg']]
            other_surg = row.iloc[LABEL_COLS['other_surg']]

            if pd.isna(main_diag) or pd.isna(main_surg):
                print(f"跳过第{idx}行（标签缺失）")
                continue

            output = format_output(main_diag, other_diag, main_surg, other_surg)
            records.append({
                'text': text,
                'main_diag': str(main_diag),
                'other_diag': str(other_diag) if pd.notna(other_diag) else '',
                'main_surg': str(main_surg),
                'other_surg': str(other_surg) if pd.notna(other_surg) else '',
                'output': output,
            })
        else:
            # 测试集只需要text和ID
            records.append({
                'text': text,
                'id': str(row.iloc[0]) if not pd.isna(row.iloc[0]) else str(idx),
            })

    return records

def create_classification_data(records):
    """转换为多标签分类格式（每个任务独立）"""
    data = []
    for rec in records:
        item = {
            'text': rec['text'],
            'main_diag': rec.get('main_diag', ''),
            'other_diag': rec.get('other_diag', ''),
            'main_surg': rec.get('main_surg', ''),
            'other_surg': rec.get('other_surg', ''),
        }
        data.append(item)
    return data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理训练集
    print("处理训练集...")
    train_df = read_excel(TRAIN_FILE)
    train_records = process_data(train_df, is_train=True)
    print(f"  有效训练样本: {len(train_records)}")

    # 保存完整训练数据
    with open(f"{OUTPUT_DIR}/train_full.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    # 分割训练/验证集 (9:1)
    import random
    random.seed(42)
    random.shuffle(train_records)
    val_size = len(train_records) // 10
    val_records = train_records[:val_size]
    train_records_split = train_records[val_size:]

    with open(f"{OUTPUT_DIR}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_records_split, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/dev.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)

    print(f"  训练: {len(train_records_split)}, 验证: {len(val_records)}")

    # 处理测试集
    if os.path.exists(TEST_FILE):
        print("处理测试集...")
        test_df = read_excel(TEST_FILE)
        test_records = process_data(test_df, is_train=False)
        print(f"  测试样本: {len(test_records)}")
        with open(f"{OUTPUT_DIR}/test.json", "w", encoding="utf-8") as f:
            json.dump(test_records, f, ensure_ascii=False, indent=2)
    else:
        print(f"测试文件不存在: {TEST_FILE}")

    # 保存类别信息
    with open(f"{OUTPUT_DIR}/classes.json", "w", encoding="utf-8") as f:
        json.dump({
            'main_diag_classes': MAIN_DIAG_CLASSES,
            'main_surg_classes': MAIN_SURG_CLASSES,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n数据保存到: {OUTPUT_DIR}")
    print("完成!")

if __name__ == "__main__":
    main()
