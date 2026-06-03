#!/usr/bin/env python3
"""分步测试模型加载"""
import sys
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch
import transformers
print(f"[1] torch={torch.__version__}, cuda={torch.cuda.is_available()}")

# Step 1: Load tokenizer with slow tokenizer fallback
from transformers import AutoTokenizer
MODEL_PATH = "/root/internlm2_5-7b-chat"

print("[2] Loading tokenizer...")
try:
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,  # force slow tokenizer
    )
    print(f"    Slow tokenizer loaded, vocab={tok.vocab_size}")
except Exception as e:
    print(f"    Slow tokenizer failed: {e}")
    print("[2b] Trying fast tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    print(f"    Fast tokenizer loaded, vocab={tok.vocab_size}")

print(f"    pad_token={tok.pad_token}, eos_token={tok.eos_token}")

# Step 2: Check model files
import glob
files = sorted(glob.glob(os.path.join(MODEL_PATH, "*.safetensors")))
print(f"[3] Model safetensor files ({len(files)}):")
for f in files:
    size = os.path.getsize(f) / 1024**3
    print(f"    {os.path.basename(f)}: {size:.2f}GB")

# Step 3: Load model config
from transformers import AutoConfig
print("[4] Loading config...")
config = AutoConfig.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
print(f"    Model type: {config.model_type}")
print(f"    Hidden size: {config.hidden_size}")
print(f"    Num params (approx): {config.hidden_size * config.num_attention_heads * config.num_hidden_layers / 1e9:.1f}B")

# Step 4: Load model weights
print("[5] Loading model weights to CPU first...")
from safetensors.torch import load_file
cpu_state = {}
for f in files:
    print(f"    Loading {os.path.basename(f)}...")
    state = load_file(f, device="cpu")
    cpu_state.update(state)
    print(f"      Loaded {len(state)} tensors")

total_params = sum(v.numel() for v in cpu_state.values())
print(f"    Total params: {total_params / 1e9:.2f}B")
total_size = sum(v.numel() * v.element_size() for v in cpu_state.values()) / 1024**3
print(f"    Total size (FP32): {total_size:.2f}GB")

# Step 5: Load model with device_map
print("[6] Loading model with device_map='auto'...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print("[6] Model loaded!")
if torch.cuda.is_available():
    print(f"    GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.max_memory_allocated()/1024**3:.2f}GB peak")

# Step 6: Test inference
print("[7] Testing inference...")
messages = [{"role": "user", "content": "你好"}]
input_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
output = tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"    Response: {output}")

print("\n[OK] All tests passed!")
