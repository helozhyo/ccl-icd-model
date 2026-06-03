#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import json
from transformers import AutoTokenizer

MODEL_PATH = '/root/autodl-tmp/models/internlm_InternLM2-1_8B'
DATA_DIR = '/root/autodl-tmp/icd_data'
max_len = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(DATA_DIR + '/train.json') as f:
    data = json.load(f)

SYSTEM = '你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...某个字段无编码时留空。'
PROMPT = '病历文本：\n{text}\n\n请严格按格式预测ICD编码（只输出编码）：'

for sample_idx in range(5):
    item = data[sample_idx]
    messages_with_response = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': PROMPT.format(text=item['text'])},
        {'role': 'assistant', 'content': item['output']},
    ]
    messages_prompt_only = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': PROMPT.format(text=item['text'])},
    ]

    # Without truncation
    full_true = tokenizer.apply_chat_template(messages_with_response, add_generation_prompt=False, return_tensors=None)
    prompt_true = tokenizer.apply_chat_template(messages_prompt_only, add_generation_prompt=True, return_tensors=None)

    # With truncation at max_len
    full_trunc = tokenizer.apply_chat_template(messages_with_response, add_generation_prompt=False, truncation=True, max_length=max_len, return_tensors=None)
    prompt_trunc = tokenizer.apply_chat_template(messages_prompt_only, add_generation_prompt=True, truncation=True, max_length=max_len, return_tensors=None)

    resp_true = len(full_true) - len(prompt_true)
    resp_trunc = len(full_trunc) - len(prompt_trunc)

    print(f"Sample {sample_idx}: full={len(full_true)} tokens (trunc={len(full_trunc)}), "
          f"prompt={len(prompt_true)} tokens (trunc={len(prompt_trunc)}), "
          f"response_true={resp_true}, response_trunc={resp_trunc}, "
          f"labels_valid={resp_trunc if resp_trunc > 0 else 0}")
