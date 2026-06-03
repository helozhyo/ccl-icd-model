#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import json
from transformers import AutoTokenizer

MODEL_PATH = '/root/autodl-tmp/models/internlm_InternLM2-1_8B'
DATA_DIR = '/root/autodl-tmp/icd_data'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(DATA_DIR + '/train.json') as f:
    data = json.load(f)

SYSTEM = '你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...某个字段无编码时留空。'
PROMPT = '病历文本：\n{text}\n\n请严格按格式预测ICD编码（只输出编码）：'

# Analyze token distribution
results = []
for item in data:
    text = item['text']
    output = item['output']
    messages_prompt_only = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': PROMPT.format(text=text)},
    ]
    messages_with_response = messages_prompt_only + [{'role': 'assistant', 'content': output}]

    prompt_tokens = len(tokenizer.apply_chat_template(messages_prompt_only, add_generation_prompt=True, return_tensors=None))
    full_tokens = len(tokenizer.apply_chat_template(messages_with_response, add_generation_prompt=False, return_tensors=None))
    resp_tokens = full_tokens - prompt_tokens
    results.append({
        'text_len': len(text),
        'prompt_tokens': prompt_tokens,
        'full_tokens': full_tokens,
        'resp_tokens': resp_tokens,
    })

print(f"Dataset: {len(results)} samples")
print(f"Text char lengths: min={min(r['text_len'] for r in results)}, max={max(r['text_len'] for r in results)}, avg={sum(r['text_len'] for r in results)/len(results):.0f}")
print(f"Prompt tokens: min={min(r['prompt_tokens'] for r in results)}, max={max(r['prompt_tokens'] for r in results)}, avg={sum(r['prompt_tokens'] for r in results)/len(results):.0f}")
print(f"Response tokens: min={min(r['resp_tokens'] for r in results)}, max={max(r['resp_tokens'] for r in results)}, avg={sum(r['resp_tokens'] for r in results)/len(results):.0f}")

# Find max text chars for different max_length targets (reserve ~80 tokens for response)
for target_max_len in [512, 768, 1024, 1536, 2048]:
    resp_reserve = 80
    usable = target_max_len - resp_reserve
    # Binary search approximate text_char limit
    low, high = 0, 5000
    for _ in range(20):
        mid = (low + high) // 2
        sample = data[0]
        truncated = sample['text'][:mid]
        msgs = [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': PROMPT.format(text=truncated)},
        ]
        t = len(tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=None))
        if t <= usable:
            low = mid
        else:
            high = mid
    print(f"max_length={target_max_len}: can fit ~{low} chars of text (prompt~{usable} tokens)")
