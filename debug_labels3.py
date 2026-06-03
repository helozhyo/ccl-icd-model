#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch, json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = '/root/autodl-tmp/models/internlm_InternLM2-1_8B'
DATA_DIR = '/root/autodl-tmp/icd_data'
max_len = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map='auto', trust_remote_code=True,
    local_files_only=True, torch_dtype=torch.float16
)
lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['wqkv', 'wo'],
                       lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_cfg)
for name, param in model.named_parameters():
    if 'lora_' not in name.lower() and 'modules_to_save' not in name.lower():
        param.requires_grad = False

with open(DATA_DIR + '/train.json') as f:
    data = json.load(f)

SYSTEM = '你是一个专业的医学编码助手。根据电子病历文本，预测ICD诊断编码和手术编码。严格按以下格式输出：主要诊断编码|其他诊断编码1;其他诊断编码2;...|主要手术编码|其他手术编码1;其他手术编码2;...某个字段无编码时留空。'
PROMPT = '病历文本：\n{text}\n\n请严格按格式预测ICD编码（只输出编码）：'

item = data[0]
print("=" * 60)
print("=== DEBUG SAMPLE 0 ===")
print("text[:100]:", repr(item['text'][:100]))
print("output:", repr(item['output']))

messages = [
    {'role': 'system', 'content': SYSTEM},
    {'role': 'user', 'content': PROMPT.format(text=item['text'])},
    {'role': 'assistant', 'content': item['output']},
]
full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print("\nfull[:200]:", repr(full[:200]))
print("full[-200:]:", repr(full[-200:]))
print("full len (chars):", len(full))

output = item['output']
aidx = full.rfind(output)
print("\naidx:", aidx)
print("full[aidx:aidx+50]:", repr(full[aidx:aidx+50]))

prompt_text = full[:aidx]
print("prompt_text len (chars):", len(prompt_text))
print("prompt_text[-100:]:", repr(prompt_text[-100:]))

# Check tokenization
enc_full_nopad = tokenizer(full, truncation=True, max_length=max_len, padding=False, return_tensors=None)
enc_prompt_nopad = tokenizer(prompt_text, truncation=True, max_length=max_len, padding=False, return_tensors=None)
print("\nenc_full_nopad tokens:", len(enc_full_nopad['input_ids']))
print("enc_prompt_nopad tokens:", len(enc_prompt_nopad['input_ids']))

# Show where they differ
full_ids = enc_full_nopad['input_ids']
prompt_ids = enc_prompt_nopad['input_ids']
for i in range(min(len(prompt_ids), len(full_ids))):
    if prompt_ids[i] != full_ids[i]:
        print(f"First diff at position {i}: prompt={prompt_ids[i]} ({tokenizer.decode([prompt_ids[i]])!r}), full={full_ids[i]} ({tokenizer.decode([full_ids[i]])!r})")
        break
else:
    print("All tokens match!")

# Try to find the output tokens in full_ids
output_tokens = tokenizer.encode(output, add_special_tokens=False)
print("\noutput_tokens:", output_tokens)
print("output_tokens decoded:", [tokenizer.decode([t]) for t in output_tokens])

# Find where output starts in full_ids
for i in range(len(full_ids) - len(output_tokens) + 1):
    if full_ids[i:i+len(output_tokens)] == output_tokens:
        print(f"Output tokens found at position {i} in full_ids")
        break
else:
    print("Output tokens NOT found in full_ids!")

# Build labels properly: use position of first output token
# Find first non-prompt position
# Strategy: find where response starts by looking for output token pattern
response_start = -1
for i in range(len(full_ids) - len(output_tokens) + 1):
    if full_ids[i:i+len(output_tokens)] == output_tokens:
        response_start = i
        break

print(f"\nresponse_start position: {response_start}")

# Build labels: -100 for prompt, token_id for response
labels_v1 = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
labels_v1 = (labels_v1 + [-100] * max_len)[:max_len]
print(f"labels_v1 (prompt_len={len(prompt_ids)}): -100={labels_v1.count(-100)}, non={sum(1 for x in labels_v1 if x!=-100)}")

# Alternative: use response_start
labels_v2 = [-100] * response_start + full_ids[response_start:]
labels_v2 = (labels_v2 + [-100] * max_len)[:max_len]
print(f"labels_v2 (response_start={response_start}): -100={labels_v2.count(-100)}, non={sum(1 for x in labels_v2 if x!=-100)}")

# Now test training with v2 labels
print("\n=== Test training with correct labels ===")
enc_full_pad = tokenizer(full, truncation=True, max_length=max_len, padding='max_length', return_tensors=None)
input_ids = torch.LongTensor([enc_full_pad['input_ids']]).cuda()
attention_mask = torch.LongTensor([enc_full_pad['attention_mask']]).cuda()
labels_t = torch.LongTensor([labels_v2]).cuda()

print("input_ids shape:", input_ids.shape)
print("labels -100:", labels_t.tolist()[0].count(-100), "non:", sum(1 for x in labels_t.tolist()[0] if x != -100))

model.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

with torch.autocast('cuda', dtype=torch.float16):
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
    loss = out.loss
print(f"loss: {loss.item():.4f}, nan: {torch.isnan(loss).sum().item()}")
