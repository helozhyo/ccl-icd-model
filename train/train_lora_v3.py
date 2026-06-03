#!/usr/bin/env python3
"""
CCL 2026 ICD - LoRA微调 (device_map + gradient checkpointing + use_reentrant=False)
通过use_reentrant=False解决CPU-GPU跨设备梯度问题
"""
import os, sys, json, warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

MODEL_PATH = "/root/internlm2_5-7b-chat"
DATA_DIR = "/root/autodl-tmp/icd_data"
OUTPUT_DIR = "/root/autodl-tmp/output/icd_lora"

CFG = {
    "max_length": 512,    # 512足够，激活显存~1.5GB
    "batch_size": 1,
    "gradient_accumulation": 1,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "seed": 42,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["wqkv", "wo"],
    "max_grad_norm": 1.0,
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


def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tokenizer
    print("=" * 50)
    print("1. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   vocab={tokenizer.vocab_size}")

    # 2. 加载模型：只把embedding和少数层放GPU，其余全CPU
    # 7B模型有32层，每层约0.48GB。只放10层在GPU = 4.8GB
    device_map = {}
    device_map["model.tok_embeddings"] = "cuda:0"
    device_map["model.norm"] = "cuda:0"
    device_map["output"] = "cuda:0"
    # 只放前10层在GPU，其余22层CPU
    for i in range(32):
        device_map[f"model.layers.{i}"] = "cuda:0" if i < 10 else "cpu"

    print("=" * 50)
    print("2. 加载模型 (前10层→GPU, 后22层→CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    gc.collect()
    if torch.cuda.is_available():
        print(f"   GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # 3. 不用gradient checkpointing，改用batch_size=1 + GA=1
    # (batch_size=1时激活显存~1GB，总显存~10GB，可以跑)
    print("   不使用gradient_checkpointing (batch_size=1够用)")

    # 4. 应用LoRA
    print("=" * 50)
    print("3. 应用LoRA...")
    lora_cfg = LoraConfig(
        r=CFG["lora_rank"], lora_alpha=CFG["lora_alpha"],
        target_modules=CFG["target_modules"],
        lora_dropout=CFG["lora_dropout"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # 只训练LoRA
    for name, param in model.named_parameters():
        if "lora_" not in name.lower() and "modules_to_save" not in name.lower():
            param.requires_grad = False

    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_p = sum(p.numel() for p in model.parameters())
    print(f"   可训练: {t:,} / {all_p:,} = {t/all_p*100:.3f}%")

    # 5. 数据
    print("=" * 50)
    print("4. 加载数据...")
    train_ds = ICDDataset(f"{DATA_DIR}/train.json", tokenizer, CFG["max_length"])
    dev_ds = ICDDataset(f"{DATA_DIR}/dev.json", tokenizer, CFG["max_length"])
    print(f"   训练: {len(train_ds)}, 验证: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=CFG["batch_size"], collate_fn=collate_fn)

    # 6. 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["learning_rate"], betas=(0.9, 0.999), weight_decay=0.01,
    )

    steps_per_epoch = len(train_loader) // CFG["gradient_accumulation"]
    num_training_steps = steps_per_epoch * CFG["num_epochs"]
    num_warmup = int(num_training_steps * CFG["warmup_ratio"])

    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / max(1, num_warmup)
        progress = float(step - num_warmup) / max(1, num_training_steps - num_warmup)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    print(f"\n总步数: {num_training_steps}, 预热: {num_warmup}")
    print("=" * 50)
    print("5. 开始训练...")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(CFG["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{CFG['num_epochs']}")

        for step, batch in pbar:
            # Move to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward + Backward
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / CFG["gradient_accumulation"]

            scaler.scale(loss).backward()

            # 累积梯度
            if (step + 1) % CFG["gradient_accumulation"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item()*CFG['gradient_accumulation']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "gpu": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB",
                })

                # 定期验证
                if global_step % 200 == 0:
                    model.eval()
                    eval_loss = 0
                    with torch.no_grad():
                        for eval_batch in tqdm(dev_loader, desc="Evaluating", leave=False):
                            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                            with autocast():
                                ev_out = model(**eval_batch)
                            eval_loss += ev_out.loss.item()
                    eval_loss /= len(dev_loader)
                    print(f"\n  Step {global_step}: eval_loss={eval_loss:.4f}, best={best_loss:.4f}")

                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        print(f"  ** 保存最佳模型 (loss={best_loss:.4f})")
                        model.save_pretrained(f"{OUTPUT_DIR}/best")
                        tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")

                    model.train()
                    gc.collect()

        print(f"\n  Epoch {epoch+1} 完成!")

    # 保存
    print("=" * 50)
    print("6. 保存模型...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"已保存: {OUTPUT_DIR}/final")
    print("训练完成!")


if __name__ == "__main__":
    main()
