#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch, json
from transformers import AutoTokenizer

MODEL_PATH = '/root/autodl-tmp/models/internlm_InternLM2-1_8B'
DATA_DIR = '/root/autodl-tmp/icd_data'
max_len = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(DATA_DIR + '/train.json') as f:
    data = json.load(f)

for sample_idx in [0, 1, 2]:
    item = data[sample_idx]
    SYSTEM = '你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...某个字段无编码时留空。'
    PROMPT = '病历文本：\n{text}\n\n请严格按格式预测ICD编码（只输出编码）：'

    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': PROMPT.format(text=item['text'])},
        {'role': 'assistant', 'content': item['output']},
    ]
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    aidx = full.rfind(item['output'])
    prompt_text = full[:aidx]

    enc_full = tokenizer(full, truncation=True, max_length=max_len, padding='max_length', return_tensors=None)
    enc_prompt = tokenizer(prompt_text, truncation=True, max_length=max_len, return_tensors=None)

    # No-padding version
    enc_full_np = tokenizer(full, truncation=True, max_length=max_len, return_tensors=None)
    enc_prompt_np = tokenizer(prompt_text, truncation=True, max_length=max_len, return_tensors=None)

    full_ids = enc_full_np['input_ids']
    prompt_ids = enc_prompt_np['input_ids']

    print(f'--- Sample {sample_idx} ---')
    print(f'Full text chars: {len(full)}, Prompt chars: {len(prompt_text)}')
    print(f'Full tokens (no pad): {len(full_ids)}, Prompt tokens (no pad): {len(prompt_ids)}')

    # Check if prompt is a prefix of full
    overlap = 0
    for a, b in zip(full_ids, prompt_ids):
        if a == b:
            overlap += 1
        else:
            break
    print(f'Token overlap (prefix match): {overlap} / {len(prompt_ids)}')

    # Build labels
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = (labels + [-100] * max_len)[:max_len]
    print(f'Labels -100: {labels.count(-100)}, non -100: {sum(1 for x in labels if x != -100)}')
    print(f'Output: {repr(item["output"])}')
    print()
