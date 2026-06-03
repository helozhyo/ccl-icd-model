#!/usr/bin/env python3
"""测试InternLM2-1.8B显存占用"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  vocab={tok.vocab_size}")

print("Loading model (device_map='auto', FP16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True,
    local_files_only=True, torch_dtype=torch.float16,
)
mem = torch.cuda.memory_allocated() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"  GPU mem: {mem:.2f}GB, peak: {peak:.2f}GB")

print("\nTest inference (seq=512)...")
inputs = tok("测试文本" * 128, return_tensors="pt", truncation=True, max_length=512, padding=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f}GB, peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
print(f"  Output: {tok.decode(outputs[0][-50:], skip_special_tokens=True)[:100]}")

print("\nTest forward + backward (seq=512, bs=1)...")
torch.cuda.reset_peak_memory_stats()
labels = torch.zeros((1, 512), dtype=torch.long, device='cuda')
inputs = tok("测试" * 128, return_tensors="pt", truncation=True, max_length=512, padding=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
loss = model(**inputs, labels=labels).loss
loss.backward()
print(f"  Forward+backward peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
mem = torch.cuda.memory_allocated() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"  GPU mem: {mem:.2f}GB, peak: {peak:.2f}GB")

print("\nTest forward + backward (seq=1024, bs=1)...")
torch.cuda.reset_peak_memory_stats()
labels = torch.zeros((1, 1024), dtype=torch.long, device='cuda')
inputs = tok("测试" * 256, return_tensors="pt", truncation=True, max_length=1024, padding=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
loss = model(**inputs, labels=labels).loss
loss.backward()
print(f"  Forward+backward peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

print("\nDone!")
