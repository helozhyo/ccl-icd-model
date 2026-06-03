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
batch_size = 1
MAX_TEXT_CHARS = 700

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
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


class DS(Dataset):
    def __init__(self, data, tokenizer, max_len, max_chars):
        self.data = data; self.tokenizer = tokenizer; self.max_len = max_len; self.max_chars = max_chars
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text'][:self.max_chars]; output = item['output']
        msgs_resp = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': PROMPT.format(text=text)}, {'role': 'assistant', 'content': output}]
        msgs_prompt = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': PROMPT.format(text=text)}]
        full_ids = self.tokenizer.apply_chat_template(msgs_resp, add_generation_prompt=False, truncation=True, max_length=self.max_len, padding='max_length', return_tensors=None)
        prompt_ids = self.tokenizer.apply_chat_template(msgs_prompt, add_generation_prompt=True, truncation=True, max_length=self.max_len, return_tensors=None)
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = (labels + [-100] * self.max_len)[:self.max_len]
        return {'input_ids': torch.LongTensor(full_ids), 'attention_mask': torch.LongTensor([1] * len(full_ids) + [0] * (self.max_len - len(full_ids))), 'labels': torch.LongTensor(labels)}


def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}


ds = DS(data, tokenizer, max_len, MAX_TEXT_CHARS)
loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=True)

# Test with lower lr and grad clip AFTER step
print("=== Test: lr=1e-5, clip after step ===")
model.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=0.01)
for i, batch in enumerate(loader):
    if i >= 10: break
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(**batch)
        loss = out.loss
    loss.backward()
    # Clip gradients BEFORE step (in-place on grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Check for NaN grads
    grad_nan = any(p.grad is not None and torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
    opt.step()
    opt.zero_grad()
    print(f'  Step {i+1}: loss={loss.item():.4f}, nan={int(torch.isnan(loss).sum().item())}, grad_nan={grad_nan}')

print("\n=== Test: lr=5e-5, no clip ===")
# Reset model
del model, opt
import gc; gc.collect(); torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16)
model = get_peft_model(model, lora_cfg)
for name, param in model.named_parameters():
    if 'lora_' not in name.lower() and 'modules_to_save' not in name.lower():
        param.requires_grad = False
model.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)

for i, batch in enumerate(loader):
    if i >= 10: break
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(**batch)
        loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    print(f'  Step {i+1}: loss={loss.item():.4f}, nan={int(torch.isnan(loss).sum().item())}')

print("\nDone!")
