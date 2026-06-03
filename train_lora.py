#!/usr/bin/env python3
"""
CCL 2026 ICD自动编码任务 - LoRA微调训练
基于InternLM2.5-7B-chat，使用纯LoRA + Gradient Checkpointing
"""
import os
import sys
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import gc

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# ========== 配置 ==========
MODEL_PATH = "/root/internlm2_5-7b-chat"
DATA_DIR = "/root/autodl-tmp/icd_data"
OUTPUT_DIR = "/root/autodl-tmp/output/icd_lora"

CFG = {
    "max_length": 1536,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "seed": 42,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["wqkv", "wo"],
}

SYSTEM = """你是一个专业的医学编码助手。根据电子病历文本，预测患者的ICD诊断编码和手术编码。
严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...
某个字段无编码时留空。"""

PROMPT = """病历文本：
{text}

请严格按格式预测ICD编码（只输出编码，不要其他内容）："""


# ========== 数据集 ==========
class ICDDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        output = item["output"]

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(text=text)},
            {"role": "assistant", "content": output},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # 找到assistant回复的起始位置
        assistant_idx = full_text.rfind(output)
        prompt_text = full_text[:assistant_idx]

        enc_full = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors=None,
        )
        enc_prompt = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_len, return_tensors=None,
        )

        input_ids = enc_full["input_ids"]
        labels = [-100] * len(enc_prompt["input_ids"]) + input_ids[len(enc_prompt["input_ids"]):]
        labels = (labels + [-100] * self.max_len)[:self.max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": enc_full["attention_mask"],
            "labels": labels,
        }


def print_trainable_params(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"可训练参数: {trainable:,} / {total:,} = {trainable/total*100:.2f}%")


def main():
    set_seed(CFG["seed"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载tokenizer
    print("=" * 60)
    print("1. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   vocab_size={tokenizer.vocab_size}, pad={tokenizer.pad_token}")

    # 2. 加载模型 (FP16, 先放CPU)
    print("=" * 60)
    print("2. 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=None,  # 不预分配设备，让Trainer处理
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"   模型加载完成!")

    # 3. 启用gradient checkpointing节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 4. 应用LoRA
    print("=" * 60)
    print("3. 应用LoRA...")
    lora_cfg = LoraConfig(
        r=CFG["lora_rank"],
        lora_alpha=CFG["lora_alpha"],
        target_modules=CFG["target_modules"],
        lora_dropout=CFG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 5. 加载数据
    print("=" * 60)
    print("4. 加载数据...")
    train_ds = ICDDataset(f"{DATA_DIR}/train.json", tokenizer, CFG["max_length"])
    dev_ds = ICDDataset(f"{DATA_DIR}/dev.json", tokenizer, CFG["max_length"])
    print(f"   训练集: {len(train_ds)}, 验证集: {len(dev_ds)}")

    # 6. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, return_tensors="pt",
    )

    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=CFG["batch_size"],
        per_device_eval_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["gradient_accumulation"],
        learning_rate=CFG["learning_rate"],
        num_train_epochs=CFG["num_epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=CFG["warmup_ratio"],
        logging_steps=10,
        save_steps=300,
        eval_steps=300,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=False,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        group_by_length=False,
        dataloader_num_workers=0,
    )

    # 8. Trainer
    print("=" * 60)
    print("5. 开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 冻结基础模型参数，只训练LoRA（peft已处理，这里再次确保）
    for name, param in model.named_parameters():
        if "lora_" not in name.lower() and "modules_to_save" not in name.lower():
            param.requires_grad = False

    print_trainable_params(model)

    # 开始训练
    trainer.train()

    # 9. 保存
    print("=" * 60)
    print("6. 保存模型...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"   已保存到: {OUTPUT_DIR}/final")
    print("训练完成!")


if __name__ == "__main__":
    main()
