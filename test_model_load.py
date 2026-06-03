#!/usr/bin/env python3
"""测试模型加载"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/internlm2_5-7b-chat"
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
print(f"Tokenizer loaded, vocab size: {tok.vocab_size}")
print(f"Pad token: {tok.pad_token}, EOS: {tok.eos_token}")

print("Loading model (FP16, device_map='auto')...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
)
print("Model loaded!")
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Test inference
print("\nTest inference...")
messages = [{"role": "user", "content": "你好，请介绍一下自己。"}]
input_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(input_text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
output = tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"Response: {output}")

del model
import gc; gc.collect(); torch.cuda.empty_cache()
print("\nAll tests passed!")
