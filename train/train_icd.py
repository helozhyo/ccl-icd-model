#!/usr/bin/env python3
"""
CCL 2026 ICD自动编码任务 - QLoRA微调训练脚本
基于InternLM2.5-7B-chat，使用4-bit量化QLoRA
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore")

# ========== 配置 ==========
MODEL_PATH = "/root/internlm2_5-7b-chat"
DATA_DIR = "/root/autodl-tmp/icd_data"
OUTPUT_DIR = "/root/autodl-tmp/output/icd_model"

CONFIG = {
    "model_name": "InternLM2.5-7B",
    "max_length": 1536,       # 最大序列长度
    "batch_size": 1,          # batch size
    "gradient_accumulation": 16,  # 梯度累积
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "seed": 42,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "quantization_bit": 4,
}

# ========== 系统提示 ==========
SYSTEM_PROMPT = """你是一个专业的医学编码助手。你的任务是根据电子病历文本，预测患者的ICD诊断编码和手术编码。
请严格按照指定格式输出，格式为：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...
如果某个字段没有对应编码，则该字段留空。
"""

PROMPT_TEMPLATE = """病历文本：
{text}

请根据以上病历文本，预测ICD编码（严格按格式输出，不要多余内容）："""

# ========== 数据集 ==========
class ICDDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, is_test=False):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        if self.is_test:
            # 测试模式：构建输入
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)},
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "id": item.get("id", str(idx)),
            }

        # 训练模式
        output_text = item["output"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)},
            {"role": "assistant", "content": output_text},
        ]

        # 使用chat template构建完整文本
        full_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # assistant回复的起始位置
        assistant_start = full_text.find(output_text, full_text.find("assistant"))
        prompt_text = full_text[:assistant_start]

        enc_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        enc_prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        input_ids = enc_full["input_ids"]
        labels = [-100] * len(enc_prompt["input_ids"]) + input_ids[len(enc_prompt["input_ids"]):]
        labels = labels[: self.max_length]

        # pad labels
        if len(labels) < self.max_length:
            labels += [-100] * (self.max_length - len(labels))

        return {
            "input_ids": input_ids,
            "attention_mask": enc_full["attention_mask"],
            "labels": labels[: self.max_length],
        }


def verify_data(data_path):
    """验证数据格式"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  数据条数: {len(data)}")
    if data:
        print(f"  样本text长度: {len(data[0]['text'])}")
        if "output" in data[0]:
            print(f"  样本output: {data[0]['output']}")
    return len(data)


def setup_model():
    """加载模型（4bit QLoRA）"""
    print(f"加载模型: {MODEL_PATH}")

    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    print("  加载模型（4bit量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )

    # 准备kbit训练
    model = prepare_model_for_kbit_training(model)

    # LoRA配置
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train():
    """主训练流程"""
    set_seed(CONFIG["seed"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 验证数据
    print("\n=== 验证数据 ===")
    train_data_path = f"{DATA_DIR}/train.json"
    dev_data_path = f"{DATA_DIR}/dev.json"
    verify_data(train_data_path)
    verify_data(dev_data_path)

    # 加载模型
    model, tokenizer = setup_model()

    # 创建数据集
    print("\n=== 创建数据集 ===")
    train_dataset = ICDDataset(train_data_path, tokenizer, CONFIG["max_length"])
    dev_dataset = ICDDataset(dev_data_path, tokenizer, CONFIG["max_length"])
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(dev_dataset)}")

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_epochs"],
        lr_scheduler_type=CONFIG["lr_scheduler"],
        warmup_ratio=CONFIG["warmup_ratio"],
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
       bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        fp16=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\n=== 开始训练 ===")
    trainer.train()

    # 保存最终模型
    print(f"\n保存模型到: {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("训练完成!")


if __name__ == "__main__":
    train()
