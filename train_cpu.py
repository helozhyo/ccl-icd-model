#!/usr/bin/env python3
"""
CCL 2026 ICD - 全CPU训练（LoRA）
CPU FP32训练，350GB RAM完全够用
预计: ~15-30分钟完成3 epoch
"""
import os, sys, json, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import gc, time

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

MODEL_PATH = "/root/internlm2_5-7b-chat"
DATA_DIR = "/root/autodl-tmp/icd_data"
OUTPUT_DIR = "/root/autodl-tmp/output/icd_lora"

CFG = {
    "max_length": 1024,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "seed": 42,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["wqkv", "wo"],
    "max_grad_norm": 1.0,
    "num_workers": 4,
}

SYSTEM = """你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。
严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...
某个字段无编码时留空。"""

PROMPT = """病历文本：
{text}

请严格按格式预测ICD编码（只输出编码）："""


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
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(text=item["text"])},
            {"role": "assistant", "content": item["output"]},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        output = item["output"]
        aidx = full_text.rfind(output)
        prompt_text = full_text[:aidx]

        enc_full = self.tokenizer(full_text, truncation=True, max_length=self.max_len,
                                  padding="max_length", return_tensors=None)
        enc_prompt = self.tokenizer(prompt_text, truncation=True, max_length=self.max_len,
                                    return_tensors=None)

        input_ids = enc_full["input_ids"]
        labels = [-100] * len(enc_prompt["input_ids"]) + input_ids[len(enc_prompt["input_ids"]):]
        labels = (labels + [-100] * self.max_len)[:self.max_len]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(enc_full["attention_mask"]),
            "labels": torch.LongTensor(labels),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def print_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_p = sum(p.numel() for p in model.parameters())
    print(f"  可训练: {t:,} / {all_p:,} = {t/all_p*100:.3f}%")


def main():
    start_time = time.time()
    torch.manual_seed(CFG["seed"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    print(f"设备: CPU")
    print(f"可用内存: {os.popen('free -h').read()}")

    # 1. Tokenizer
    print("\n" + "=" * 50)
    print("1. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   vocab={tokenizer.vocab_size}")

    # 2. 加载模型到CPU
    print("\n" + "=" * 50)
    print("2. 加载模型 (CPU, FP32)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=None,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    print(f"   模型加载完成! 耗时: {time.time()-t0:.1f}s")

    # 3. 应用LoRA
    print("\n" + "=" * 50)
    print("3. 应用LoRA...")
    lora_cfg = LoraConfig(
        r=CFG["lora_rank"], lora_alpha=CFG["lora_alpha"],
        target_modules=CFG["target_modules"],
        lora_dropout=CFG["lora_dropout"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    print_trainable(model)

    # 只训练LoRA
    for name, param in model.named_parameters():
        if "lora_" not in name.lower() and "modules_to_save" not in name.lower():
            param.requires_grad = False

    print_trainable(model)

    # 4. 数据
    print("\n" + "=" * 50)
    print("4. 加载数据...")
    train_ds = ICDDataset(f"{DATA_DIR}/train.json", tokenizer, CFG["max_length"])
    dev_ds = ICDDataset(f"{DATA_DIR}/dev.json", tokenizer, CFG["max_length"])
    print(f"   训练: {len(train_ds)}, 验证: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=False)
    dev_loader = DataLoader(dev_ds, batch_size=CFG["batch_size"],
                            collate_fn=collate_fn, num_workers=CFG["num_workers"])

    # 5. 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["learning_rate"], betas=(0.9, 0.999), weight_decay=0.01,
    )

    steps_per_epoch = len(train_loader) // CFG["gradient_accumulation"]
    num_training_steps = steps_per_epoch * CFG["num_epochs"]
    num_warmup = int(num_training_steps * CFG["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training_steps)

    print(f"\n总步数: {num_training_steps}, 预热: {num_warmup}")
    print(f"每epoch: {steps_per_epoch} 步, 每步有效batch: {CFG['batch_size'] * CFG['gradient_accumulation']}")

    # 6. 训练
    print("\n" + "=" * 50)
    print("5. 开始训练...")
    print(f"   开始时间: {time.strftime('%H:%M:%S')}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(CFG["num_epochs"]):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{CFG['num_epochs']}", ncols=80)

        for step, batch in pbar:
            outputs = model(**batch)
            loss = outputs.loss / CFG["gradient_accumulation"]
            loss.backward()

            if (step + 1) % CFG["gradient_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item()*CFG['gradient_accumulation']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # 定期验证
            if global_step > 0 and global_step % 200 == 0:
                model.eval()
                eval_loss = 0
                eval_count = 0
                with torch.no_grad():
                    for eval_batch in tqdm(dev_loader, desc="Evaluating", leave=False, ncols=80):
                        ev_out = model(**eval_batch)
                        eval_loss += ev_out.loss.item()
                        eval_count += 1
                eval_loss /= max(eval_count, 1)
                elapsed = time.time() - start_time
                print(f"\n  Step {global_step}: eval_loss={eval_loss:.4f} | "
                      f"累计时间: {elapsed/60:.1f}min | 速度: {global_step/elapsed:.1f} step/s")

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    print(f"  ** 保存最佳模型 (loss={best_loss:.4f})")
                    model.save_pretrained(f"{OUTPUT_DIR}/best")
                    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")

                model.train()

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1} 完成! 耗时: {epoch_time/60:.1f}min")

    # 保存
    print("\n" + "=" * 50)
    print("6. 保存模型...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    total_time = time.time() - start_time
    print(f"已保存: {OUTPUT_DIR}/final")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print("训练完成!")


if __name__ == "__main__":
    main()
