#!/usr/bin/env python3
"""
CCL 2026 ICD自动编码 - 推理脚本
对A榜测试集进行预测，输出结果到CSV
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

# ========== 配置 ==========
MODEL_PATH = "/root/internlm2_5-7b-chat"
LORA_PATH = "/root/autodl-tmp/output/icd_model/final"
DATA_DIR = "/root/autodl-tmp/icd_data"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
OUTPUT_FILE = "/root/autodl-tmp/predictions.csv"

CONFIG = {
    "max_length": 1536,
    "batch_size": 1,
    "temperature": 0.01,
    "top_p": 0.9,
    "max_new_tokens": 256,
}

SYSTEM_PROMPT = """你是一个专业的医学编码助手。你的任务是根据电子病历文本，预测患者的ICD诊断编码和手术编码。
请严格按照指定格式输出，格式为：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...
如果某个字段没有对应编码，则该字段留空。
"""

PROMPT_TEMPLATE = """病历文本：
{text}

请根据以上病历文本，预测ICD编码（严格按格式输出，不要多余内容）："""


def setup_model():
    """加载模型"""
    print(f"加载基础模型: {MODEL_PATH}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )

    # 加载LoRA权重
    if os.path.exists(LORA_PATH):
        print(f"  加载LoRA: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH, is_trainable=False)
        model.eval()
    else:
        print(f"  LoRA路径不存在: {LORA_PATH}，使用基础模型")

    return model, tokenizer


def read_test_data():
    """读取测试数据"""
    import openpyxl
    xl = pd.ExcelFile(TEST_FILE, engine='openpyxl')
    df = xl.parse(xl.sheet_names[0])
    return df


def predict_batch(model, tokenizer, texts, max_new_tokens=256):
    """批量预测"""
    results = []
    for text in tqdm(texts, desc="预测中"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"])
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_ids = outputs[0][len(inputs["input_ids"][0]):]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        results.append(output_text)

    return results


def parse_output(output_text):
    """
    解析模型输出，按格式: 主要诊断|其他诊断;...|主要手术|其他手术;...
    """
    output_text = output_text.strip()

    # 尝试提取四个字段
    # 先按|分割，但注意其他诊断/手术中可能有|
    parts = output_text.split("|")
    if len(parts) >= 4:
        main_diag = parts[0].strip()
        other_diag = parts[1].strip()
        main_surg = parts[2].strip()
        other_surg = parts[3].strip()
    elif len(parts) == 3:
        main_diag = parts[0].strip()
        other_diag = parts[1].strip()
        main_surg = parts[2].strip()
        other_surg = ""
    elif len(parts) == 2:
        main_diag = parts[0].strip()
        other_diag = ""
        main_surg = parts[1].strip()
        other_surg = ""
    else:
        main_diag = output_text.strip()
        other_diag = ""
        main_surg = ""
        other_surg = ""

    return main_diag, other_diag, main_surg, other_surg


def main():
    print("=== ICD自动编码推理 ===\n")

    # 加载模型
    model, tokenizer = setup_model()

    # 读取测试数据
    print("\n读取测试数据...")
    test_df = read_test_data()
    print(f"  测试样本数: {len(test_df)}")

    # 提取文本字段 (col 1-14)
    TEXT_COLS = list(range(1, 15))
    texts = []
    ids = []

    for idx, row in test_df.iterrows():
        parts = []
        for col_idx in TEXT_COLS:
            val = row.iloc[col_idx]
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        texts.append('\n'.join(parts))
        ids.append(str(row.iloc[0]) if pd.notna(row.iloc[0]) else str(idx))

    # 预测
    print(f"\n开始预测 {len(texts)} 条样本...")
    raw_outputs = predict_batch(model, tokenizer, texts)

    # 解析并保存
    results = []
    for id_, raw in zip(ids, raw_outputs):
        main_d, other_d, main_s, other_s = parse_output(raw)
        results.append({
            "病案标识": id_,
            "主要诊断编码": main_d,
            "其他诊断编码": other_d,
            "主要手术编码": main_s,
            "其他手术编码": other_s,
            "raw_output": raw,
        })

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存: {OUTPUT_FILE}")
    print("\n样例预测:")
    for r in results[:3]:
        print(f"  ID={r['病案标识']}: {r['主要诊断编码']}|{r['其他诊断编码']}|{r['主要手术编码']}|{r['其他手术编码']}")


if __name__ == "__main__":
    main()
