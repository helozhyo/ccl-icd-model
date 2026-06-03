#!/usr/bin/env python3
"""
InternLM2-1_8B + LoRA 推理脚本
预测A_test.xlsx中的ICD编码
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

SYSTEM = """你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。
严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...
某个字段无编码时留空。"""

PROMPT = """病历文本：
{text}

请严格按格式预测ICD编码（只输出编码）："""

BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
LORA_PATH = "/root/autodl-tmp/output/icd_lora_1b8/best"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
OUTPUT_FILE = "/root/autodl-tmp/predictions_1b8.csv"


def concat_text(row):
    """将病历的14个文本列拼接"""
    parts = []
    for col_idx in range(1, 15):
        val = row.iloc[col_idx]
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return '\n'.join(parts)


def parse_output(output_text):
    """解析模型输出，分割4个字段"""
    parts = output_text.split("|")
    main_d = parts[0].strip() if len(parts) > 0 else ""
    other_d = parts[1].strip() if len(parts) > 1 else ""
    main_s = parts[2].strip() if len(parts) > 2 else ""
    other_s = parts[3].strip() if len(parts) > 3 else ""
    return main_d, other_d, main_s, other_s


def main():
    print("=" * 60)
    print("加载模型...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )

    print("加载LoRA权重...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH, device_map="auto")
    model.eval()

    if torch.cuda.is_available():
        print(f"GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # 读取测试数据
    print("读取测试数据...")
    df = pd.read_excel(TEST_FILE, engine='openpyxl')
    print(f"样本数: {len(df)}")

    texts = []
    ids = []
    for idx, row in df.iterrows():
        texts.append(concat_text(row))
        ids.append(str(row.iloc[0]) if pd.notna(row.iloc[0]) else str(idx))

    # 预测
    results = []
    for i, (id_, text) in enumerate(tqdm(list(zip(ids, texts)), desc="推理")):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(text=text[:3000])},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.1,
            )

        output_ids = outputs[0][len(inputs["input_ids"][0]):]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        main_d, other_d, main_s, other_s = parse_output(output_text)

        print(f"\n[{i+1}/{len(texts)}] ID={id_}")
        print(f"  预测: {main_d}|{other_d[:60] if other_d else ''}|{main_s}|{other_s[:40] if other_s else ''}")

        results.append({
            "病案标识": id_,
            "主要诊断编码": main_d,
            "其他诊断编码": other_d,
            "主要手术编码": main_s,
            "其他手术编码": other_s,
        })

    # 保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {OUTPUT_FILE}")
    print(f"共预测 {len(results)} 条记录")


if __name__ == "__main__":
    main()
