#!/usr/bin/env python3
import torch, os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from transformers import AutoModel

MODEL_PATH = "/root/internlm2_5-7b-chat"
print("Loading model (cpu)...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
)
print("=" * 40)
print("All named_children:")
for name, child in model.named_children():
    print(f"  {name}: {type(child).__name__}")

inner = model.model
print("\nnamed_children of model.model:")
for name, child in inner.named_children():
    print(f"  model.model.{name}: {type(child).__name__}")

print("\nnamed_parameters (first 10):")
for name, p in list(model.named_parameters())[:10]:
    print(f"  {name}: {p.shape}")

print("\nAll attention module paths:")
for name, m in model.named_modules():
    if "attention" in name.lower() and ("wqkv" in name.lower() or "wq" in name.lower()):
        print(f"  {name}")
