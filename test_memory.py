#!/usr/bin/env python3
"""测试基础模型显存占用"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/internlm2_5-7b-chat"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading model with device_map='auto'...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True,
    local_files_only=True, torch_dtype=torch.float16,
)
print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print(f"Peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

# Test forward
print("\nTest forward pass (seq_len=512)...")
inputs = tok("测试文本", return_tensors="pt", padding=True, max_length=512, truncation=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
print(f"Input size: {inputs['input_ids'].shape}")
with torch.cuda.amp.autocast():
    outputs = model(**inputs)
print(f"After forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB, peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

print("\nTest forward pass (seq_len=1024)...")
inputs = tok("测试" * 256, return_tensors="pt", padding=True, max_length=1024, truncation=True)
inputs = {k: v.cuda() for k, v in inputs.items()}
print(f"Input size: {inputs['input_ids'].shape}")
with torch.cuda.amp.autocast():
    outputs = model(**inputs)
print(f"After forward: {torch.cuda.memory_allocated()/1024**3:.2f}GB, peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

print("\nTest backward pass...")
labels = torch.zeros((1, 1024), dtype=torch.long, device='cuda')
loss = model(**inputs, labels=labels).loss
print(f"Loss: {loss.item():.4f}")
loss.backward()
print(f"After backward: {torch.cuda.memory_allocated()/1024**3:.2f}GB, peak: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

print("\nDone!")
