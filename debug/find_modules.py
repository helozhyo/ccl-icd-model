#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/miniconda3/lib/python3.10/site-packages')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/internlm2_5-7b-chat"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
)
print("Finding attention modules...")
for name, module in model.named_modules():
    # Look for attention-related modules
    if any(k in name.lower() for k in ['attn', 'qkv', 'query', 'value', 'key']):
        print(f"  {name}: {type(module).__name__}")
print("\nFinding linear layers in transformer blocks...")
linear_layers = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if name not in linear_layers:
            linear_layers[name] = 0
        linear_layers[name] += 1

# Print only in transformer.h layers
for name in sorted(linear_layers.keys()):
    if 'transformer.h' in name or 'model.layers' in name or 'decoder.layers' in name:
        print(f"  {name}")
