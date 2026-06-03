#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/miniconda3/lib/python3.10/site-packages')

import transformers
print(f"transformers={transformers.__version__}")

try:
    from transformers.cache_utils import Cache, DynamicCache
    print("cache_utils OK")
except ImportError as e:
    print(f"cache_utils FAILED: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    MODEL_PATH = "/root/internlm2_5-7b-chat"
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print(f"Tokenizer OK, vocab={tok.vocab_size}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    print("Model loaded!")
    if torch.cuda.is_available():
        print(f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    # Test inference
    messages = [{"role": "user", "content": "你好"}]
    input_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    output = tok.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"Response: {output}")

    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
