#!/usr/bin/env python3
"""
在dev.json上推理并评测（dev.json有标签）
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import json, torch, gc
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import f1_score

BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
LORA_PATH = "/root/autodl-tmp/output/icd_lora_1b8/best"
DEV_FILE = "/root/autodl-tmp/icd_data/dev.json"

SYSTEM = "你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...某个字段无编码时留空。"
PROMPT = "病历文本：\n{text}\n\n请严格按格式预测ICD编码（只输出编码）："
MAX_TEXT_CHARS = 700


def parse_output(output_text):
    """解析模型输出"""
    # 移除特殊token
    output_text = output_text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    parts = output_text.split("|")
    main_d = parts[0].strip() if len(parts) > 0 else ""
    other_d = parts[1].strip() if len(parts) > 1 else ""
    main_s = parts[2].strip() if len(parts) > 2 else ""
    other_s = parts[3].strip() if len(parts) > 3 else ""
    return main_d, other_d, main_s, other_s


def safe_f1(y_true_list, y_pred_list):
    """计算F1，处理空列表"""
    valid = [(t, p) for t, p in zip(y_true_list, y_pred_list) if t or p]
    if not valid:
        return 0.0
    y_t = [set(x.split(';')) if x else set() for x, _ in valid]
    y_p = [set(x.split(';')) if x else set() for _, x in valid]
    all_labels = set()
    for s in y_t + y_p:
        all_labels.update(s)
    all_labels = sorted(all_labels)
    if not all_labels:
        return 0.0
    y_t_bin = [[1 if lab in s else 0 for lab in all_labels] for s in y_t]
    y_p_bin = [[1 if lab in s else 0 for lab in all_labels] for s in y_p]
    return f1_score(y_t_bin, y_p_bin, average='micro', zero_division=0)


def main():
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH, device_map="auto")
    model.eval()
    print(f"GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # 加载dev数据（带标签）
    with open(DEV_FILE, 'r') as f:
        dev_data = json.load(f)
    print(f"Dev样本: {len(dev_data)}")

    # 推理
    results = []
    for i, item in enumerate(tqdm(dev_data, desc="推理")):
        text = item["text"][:MAX_TEXT_CHARS]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(text=text)},
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        output_ids = outputs[0][len(inputs["input_ids"][0]):]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        main_d, other_d, main_s, other_s = parse_output(output_text)
        results.append({
            "idx": i,
            "main_d": main_d,
            "other_d": other_d,
            "main_s": main_s,
            "other_s": other_s,
            "label_main_d": item.get("main_diag", ""),
            "label_other_d": item.get("other_diag", ""),
            "label_main_s": item.get("main_surg", ""),
            "label_other_s": item.get("other_surg", ""),
        })

        if i < 3:
            print(f"\n[{i}] 预测: {main_d}|{other_d[:50]}|{main_s}|{other_s[:30]}")
            print(f"    标签: {item.get('main_diag','')}|{item.get('other_diag','')}|{item.get('main_surg','')}|{item.get('other_surg','')}")

    # 评测
    print("\n" + "=" * 50)
    print("评测结果:")

    pred_main_d = [r["main_d"] for r in results]
    pred_other_d = [r["other_d"] for r in results]
    pred_main_s = [r["main_s"] for r in results]
    pred_other_s = [r["other_s"] for r in results]

    label_main_d = [r["label_main_d"] for r in results]
    label_other_d = [r["label_other_d"] for r in results]
    label_main_s = [r["label_main_s"] for r in results]
    label_other_s = [r["label_other_s"] for r in results]

    # Acc_main
    acc_main = sum(1 for p, l in zip(pred_main_d, label_main_d) if p.strip() == l.strip()) / len(results)
    print(f"  Acc_main: {acc_main:.4f}")

    # F1_other_diag
    f1_other_d = safe_f1(label_other_d, pred_other_d)
    print(f"  F1_other_diag: {f1_other_d:.4f}")

    # Acc_main_surg
    acc_main_surg = sum(1 for p, l in zip(pred_main_s, label_main_s) if p.strip() == l.strip()) / len(results)
    print(f"  Acc_main_surg: {acc_main_surg:.4f}")

    # F1_other_surg
    f1_other_s = safe_f1(label_other_s, pred_other_s)
    print(f"  F1_other_surg: {f1_other_s:.4f}")

    # M_total
    M_total = 0.4 * acc_main + 0.1 * f1_other_d + 0.4 * acc_main_surg + 0.1 * f1_other_s
    print(f"\n  M_total = 0.4*{acc_main:.4f} + 0.1*{f1_other_d:.4f} + 0.4*{acc_main_surg:.4f} + 0.1*{f1_other_s:.4f}")
    print(f"  M_total = {M_total:.4f}")

    # 保存详细结果
    result_df = pd.DataFrame(results)
    result_df.to_csv("/root/autodl-tmp/dev_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存: /root/autodl-tmp/dev_predictions.csv")


if __name__ == "__main__":
    main()
