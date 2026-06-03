#!/usr/bin/env python3
"""
CCL 2026 ICD - LoRA微调 (InternLM2-1_8B)
修复：enc_prompt 不截断，用实际token数定位label边界
"""
import os, json, warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

MODEL_PATH = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
DATA_DIR = "/root/autodl-tmp/icd_data"
OUTPUT_DIR = "/root/autodl-tmp/output/icd_lora_1b8"

CFG = {
    "max_length": 768,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "seed": 42,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["wqkv", "wo"],
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_steps": 200,
}

SYSTEM = "你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...某个字段无编码时留空。"
PROMPT = "病历文本：\n{text}\n\n请严格按格式预测ICD编码（只输出编码）："


class ICDDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 检查数据
        item = self.data[0]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(text=item["text"])},
            {"role": "assistant", "content": item["output"]},
        ]
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        aidx = full.rfind(item["output"])
        prompt_text = full[:aidx]
        enc_p = tokenizer(prompt_text, truncation=False, return_tensors=None)  # 不截断
        enc_f = tokenizer(full, truncation=True, max_length=max_len, return_tensors=None)
        print(f"  Prompt tokens: {len(enc_p['input_ids'])}, Full tokens (trunc): {len(enc_f['input_ids'])}")

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

        # 关键：enc_prompt 不截断，获取完整prompt token数
        enc_prompt = self.tokenizer(prompt_text, truncation=False, return_tensors=None)
        prompt_len = len(enc_prompt["input_ids"])

        # enc_full 截断到max_len，padding到max_len
        enc_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = enc_full["input_ids"]
        # label: prompt部分=-100, response部分=token_id
        labels = [-100] * prompt_len + input_ids[prompt_len:]
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


def evaluate(model, dev_loader, device):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(**batch)
            total_loss += out.loss.item() * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
    model.train()
    return total_loss / max(n, 1)


def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tokenizer
    print("=" * 60)
    print("1. 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   vocab={tokenizer.vocab_size}")

    # 2. Model
    print("=" * 60)
    print("2. 加载模型 (device_map='auto', FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    gc.collect()
    if torch.cuda.is_available():
        print(f"   GPU mem after load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # 3. LoRA
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

    for name, param in model.named_parameters():
        if "lora_" not in name.lower() and "modules_to_save" not in name.lower():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable:,} / {total:,} = {trainable/total*100:.3f}%")

    # 4. Data
    print("=" * 60)
    print("4. 加载数据...")
    train_ds = ICDDataset(f"{DATA_DIR}/train.json", tokenizer, CFG["max_length"])
    dev_ds = ICDDataset(f"{DATA_DIR}/dev.json", tokenizer, CFG["max_length"])
    print(f"   训练: {len(train_ds)}, 验证: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=CFG["batch_size"],
                            collate_fn=collate_fn)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=CFG["weight_decay"],
    )

    num_training_steps = len(train_loader) * CFG["num_epochs"]
    num_warmup_steps = int(num_training_steps * CFG["warmup_ratio"])

    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n总步数: {num_training_steps}, 预热: {num_warmup_steps}")

    # 6. Training
    print("=" * 60)
    print("5. 开始训练...")
    global_step = 0
    best_loss = float("inf")
    model.train()

    for epoch in range(CFG["num_epochs"]):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{CFG['num_epochs']}")

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % CFG["logging_steps"] == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "gpu": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB",
                })

            if global_step % CFG["eval_steps"] == 0 and global_step > 0:
                eval_loss = evaluate(model, dev_loader, device)
                print(f"\n  Step {global_step}: eval_loss={eval_loss:.4f}, best={best_loss:.4f}")
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    print(f"  ** 保存最佳模型 (loss={best_loss:.4f})")
                    model.save_pretrained(f"{OUTPUT_DIR}/best")
                    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best")
                gc.collect()

        print(f"\n  Epoch {epoch+1} 完成!")

    # 7. Save
    print("=" * 60)
    print("6. 保存最终模型...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"已保存: {OUTPUT_DIR}/final")
    print("训练完成!")


if __name__ == "__main__":
    main()
