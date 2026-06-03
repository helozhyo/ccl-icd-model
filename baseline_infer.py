#!/usr/bin/env python3
"""
快速推理测试：看原始InternLM2.7B在无微调情况下的ICD编码效果
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

SYSTEM = """你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。
严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...
某个字段无编码时留空。"""

PROMPT = """病历文本：
{text}

请严格按格式预测ICD编码（只输出编码）："""

MODEL_PATH = "/root/internlm2_5-7b-chat"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
OUTPUT_FILE = "/root/autodl-tmp/baseline_predictions.csv"

def concat_text(row):
    parts = []
    for col_idx in range(1, 15):
        val = row.iloc[col_idx]
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return '\n'.join(parts)

def main():
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float16,
    )
    model.eval()
    print(f"模型加载完成，GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # 读取测试数据
    print("读取测试数据...")
    df = pd.read_excel(TEST_FILE, engine='openpyxl')
    print(f"样本数: {len(df)}")

    texts = []
    ids = []
    for idx, row in df.iterrows():
        texts.append(concat_text(row))
        ids.append(str(row.iloc[0]) if pd.notna(row.iloc[0]) else str(idx))

    # 预测（只预测前20条做测试）
    results = []
    for i, (id_, text) in enumerate(tqdm(list(zip(ids, texts))[:20], desc="推理")):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(text=text[:3000])},
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        output_ids = outputs[0][len(inputs["input_ids"][0]):]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        parts = output_text.split("|")
        main_d = parts[0].strip() if len(parts) > 0 else ""
        other_d = parts[1].strip() if len(parts) > 1 else ""
        main_s = parts[2].strip() if len(parts) > 2 else ""
        other_s = parts[3].strip() if len(parts) > 3 else ""

        print(f"\n[{i+1}] ID={id_}")
        print(f"  预测: {main_d}|{other_d[:50]}|{main_s}|{other_s[:30]}")
        results.append({"病案标识": id_, "主要诊断编码": main_d,
                        "其他诊断编码": other_d, "主要手术编码": main_s, "其他手术编码": other_s})

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {OUTPUT_FILE}")
    print("基线推理完成!")


if __name__ == "__main__":
    main()
