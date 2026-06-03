#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/miniconda3/lib/python3.10/site-packages')

import torch
import transformers
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"gpu_mem={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Now test model loading
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/internlm2_5-7b-chat"
print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
print(f"Tokenizer loaded, vocab={tok.vocab_size}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
)
print("Model loaded!")
print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Test inference
print("\nTest inference...")
messages = [{"role": "user", "content": "你好"}]
input_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(input_text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
output = tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"Response: {output}")

print("\nAll tests passed!")
