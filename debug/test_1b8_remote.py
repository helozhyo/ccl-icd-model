#!/usr/bin/env python3
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading model (FP16, device_map=auto)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True,
    local_files_only=True, torch_dtype=torch.float16,
)
mem = torch.cuda.memory_allocated() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"  After load: GPU mem={mem:.2f}GB, peak={peak:.2f}GB")

# Test seq=512 forward
print("\nTest forward (seq=512)...")
inputs = tok("测试" * 128, return_tensors="pt", truncation=True, max_length=512, padding=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(f"  GPU mem={torch.cuda.memory_allocated()/1024**3:.2f}GB, peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

# Test seq=512 backward
print("\nTest forward+backward (seq=512, bs=1)...")
torch.cuda.reset_peak_memory_stats()
labels = torch.zeros((1, 512), dtype=torch.long, device="cuda")
inputs512 = tok("测试" * 128, return_tensors="pt", truncation=True, max_length=512, padding=True)
inputs512 = {k: v.cuda() for k, v in inputs512.items()}
loss = model(**inputs512, labels=labels).loss
loss.backward()
print(f"  Forward+backward peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
print(f"  GPU mem={torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Test seq=1024 backward
print("\nTest forward+backward (seq=1024, bs=1)...")
torch.cuda.reset_peak_memory_stats()
labels = torch.zeros((1, 1024), dtype=torch.long, device="cuda")
inputs1024 = tok("测试" * 256, return_tensors="pt", truncation=True, max_length=1024, padding=True)
inputs1024 = {k: v.cuda() for k, v in inputs1024.items()}
loss = model(**inputs1024, labels=labels).loss
loss.backward()
print(f"  Forward+backward peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
print("\nDone!")
