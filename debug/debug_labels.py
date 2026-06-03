#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch, json
from transformers import AutoTokenizer

MODEL_PATH = '/root/autodl-tmp/models/internlm_InternLM2-1_8B'
DATA_DIR = '/root/autodl-tmp/icd_data'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('pad_token_id:', tokenizer.pad_token_id)
print('eos_token_id:', tokenizer.eos_token_id)

with open(DATA_DIR + '/train.json', 'r') as f:
    data = json.load(f)
item = data[0]
print('text len:', len(item['text']))
print('output:', repr(item['output']))

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
print('Full len:', len(full), 'Prompt len:', len(prompt_text))

enc_full = tokenizer(full, truncation=True, max_length=512, padding='max_length', return_tensors=None)
enc_prompt = tokenizer(prompt_text, truncation=True, max_length=512, return_tensors=None)
print('Full tokens:', len(enc_full['input_ids']), 'Prompt tokens:', len(enc_prompt['input_ids']))

input_ids = enc_full['input_ids']
labels = [-100] * len(enc_prompt['input_ids']) + input_ids[len(enc_prompt['input_ids']):]
labels = (labels + [-100] * 512)[:512]
print('labels -100 count:', labels.count(-100))
print('labels non -100 count:', sum(1 for x in labels if x != -100))

pad_id = tokenizer.pad_token_id
print('pad_id:', pad_id, 'pad in labels:', labels.count(pad_id))

# Show sample labels
non_pad = [i for i, x in enumerate(labels) if x != -100 and x != pad_id]
if non_pad:
    print('First 5 label IDs:', labels[non_pad[:5]])
    print('First 5 decoded:', [tokenizer.decode([labels[i]]) for i in non_pad[:5]])

# Test forward with real data
print('\nTest forward with first sample...')
model_name = MODEL_PATH
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map='auto', trust_remote_code=True,
    local_files_only=True, torch_dtype=torch.float16,
)
model.train()

input_ids_t = torch.LongTensor([enc_full['input_ids']]).cuda()
attention_mask_t = torch.LongTensor([enc_full['attention_mask']]).cuda()
labels_t = torch.LongTensor([labels]).cuda()
print('input_ids shape:', input_ids_t.shape, 'labels shape:', labels_t.shape)

with torch.autocast('cuda', dtype=torch.float16):
    out = model(input_ids=input_ids_t, attention_mask=attention_mask_t, labels=labels_t)
    print('loss:', out.loss.item())
    print('logits shape:', out.logits.shape)

# Check logits for pad tokens
logits = out.logits[0]  # [seq, vocab]
pad_logits = logits[labels_t[0] == pad_id]
print('Pad logits mean:', pad_logits.mean().item() if pad_logits.numel() > 0 else 'N/A')
print('Non-pad logits mean:', logits[labels_t[0] != pad_id].mean().item())
