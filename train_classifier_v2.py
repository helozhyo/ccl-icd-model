"""
v2: 用 train_full.json (14401 样本) 从 best_classifier.pt 继续微调。
LR=2e-4, 3 epochs, 保存新 best 到 best_classifier_v2.pt
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
TRAIN_FILE = "/root/autodl-tmp/icd_data/train_full.json"
DEV_FILE = "/root/autodl-tmp/icd_data/dev.json"
CHECKPOINT_IN = "/root/autodl-tmp/classifier_output/best_classifier.pt"
CHECKPOINT_OUT = "/root/autodl-tmp/classifier_output/best_classifier_v2.pt"

MAX_LEN = 512
BATCH_SIZE = 4
EMBED_DIM = 512
LR = 2e-4
EPOCHS = 3
device = torch.device("cuda")


class ICDClassifier(nn.Module):
    def __init__(self, llm, embed_dim=512, num_main_diag=18, num_main_surg=16,
                 num_other_diag=1, num_other_surg=1):
        super().__init__()
        self.llm = llm
        self.hidden_dim = llm.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.mlp_dis_diag = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1))
        self.cls_main_diag = nn.Linear(embed_dim // 2, num_main_diag)
        self.mlp_dis_surg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1))
        self.cls_main_surg = nn.Linear(embed_dim // 2, num_main_surg)
        self.mlp_gen_diag = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.cls_other_diag = nn.Linear(embed_dim, num_other_diag)
        self.mlp_gen_surg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.cls_other_surg = nn.Linear(embed_dim, num_other_surg)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        e_llm = (hidden.float() * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        e_fuse = self.proj(e_llm)
        p_main_d = self.cls_main_diag(self.mlp_dis_diag(e_fuse))
        p_main_s = self.cls_main_surg(self.mlp_dis_surg(e_fuse))
        p_other_d = self.cls_other_diag(self.mlp_gen_diag(e_fuse))
        p_other_s = self.cls_other_surg(self.mlp_gen_surg(e_fuse))
        return p_main_d, p_main_s, p_other_d, p_other_s


def parse_codes(s):
    if not s:
        return []
    return [c.strip() for c in str(s).replace('；', ';').split(';') if c.strip()]


class ICDDataset(Dataset):
    def __init__(self, data, tokenizer, main_diag2idx, main_surg2idx,
                 other_diag2idx, other_surg2idx, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.main_diag2idx = main_diag2idx
        self.main_surg2idx = main_surg2idx
        self.other_diag2idx = other_diag2idx
        self.other_surg2idx = other_surg2idx
        self.max_len = max_len
        self.nd = len(other_diag2idx)
        self.ns = len(other_surg2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')[:1500]
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True,
            max_length=self.max_len, padding='max_length')
        label_d = self.main_diag2idx.get(item.get('main_diag', ''), 0)
        label_s = self.main_surg2idx.get(item.get('main_surg', ''), 0)
        od = torch.zeros(self.nd)
        for c in parse_codes(item.get('other_diag', '')):
            if c in self.other_diag2idx:
                od[self.other_diag2idx[c]] = 1.0
        os_ = torch.zeros(self.ns)
        for c in parse_codes(item.get('other_surg', '')):
            if c in self.other_surg2idx:
                os_[self.other_surg2idx[c]] = 1.0
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label_d': torch.tensor(label_d, dtype=torch.long),
            'label_s': torch.tensor(label_s, dtype=torch.long),
            'od': od,
            'os': os_,
        }


def eval_model(model, loader, main_diag_classes, main_surg_classes,
               other_diag2idx, other_surg2idx, dev_data):
    model.eval()
    all_pd, all_ps, all_od, all_os = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            iids = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            pd, ps, od, os_ = model(iids, amask)
            all_pd.append(pd.cpu()); all_ps.append(ps.cpu())
            all_od.append(torch.sigmoid(od).cpu())
            all_os.append(torch.sigmoid(os_).cpu())
    p_d = torch.cat(all_pd).numpy()
    p_s = torch.cat(all_ps).numpy()
    od = torch.cat(all_od).numpy()
    os_ = torch.cat(all_os).numpy()

    gt_md = np.array([d['main_diag'] for d in dev_data])
    gt_ms = np.array([d['main_surg'] for d in dev_data])
    nd = len(other_diag2idx)
    ns = len(other_surg2idx)
    gt_od = np.zeros((len(dev_data), nd))
    gt_os = np.zeros((len(dev_data), ns))
    for i, d in enumerate(dev_data):
        for c in parse_codes(d.get('other_diag', '')):
            if c in other_diag2idx:
                gt_od[i, other_diag2idx[c]] = 1.0
        for c in parse_codes(d.get('other_surg', '')):
            if c in other_surg2idx:
                gt_os[i, other_surg2idx[c]] = 1.0

    pred_md = np.array([main_diag_classes[p.argmax()] for p in p_d])
    pred_ms = np.array([main_surg_classes[p.argmax()] for p in p_s])
    acc_d = (pred_md == gt_md).mean()
    acc_s = (pred_ms == gt_ms).mean()

    # threshold search on dev
    best_m = 0
    best_params = (0.25, 0.4, 3)
    for td in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for ts in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for ks in [2, 3, 4]:
                f1d, f1s = [], []
                for i in range(len(dev_data)):
                    pred = set(np.where(od[i] > td)[0])
                    true = set(np.where(gt_od[i] > 0)[0])
                    if not pred and not true: f1d.append(1.0)
                    elif not pred or not true: f1d.append(0.0)
                    else:
                        tp = len(pred & true)
                        f1d.append(2*tp/(len(pred)+len(true)))
                    top = os_[i].argsort()[-ks:][::-1]
                    pred_s = set(j for j in top if os_[i][j] > ts)
                    true_s = set(np.where(gt_os[i] > 0)[0])
                    if not pred_s and not true_s: f1s.append(1.0)
                    elif not pred_s or not true_s: f1s.append(0.0)
                    else:
                        tp = len(pred_s & true_s)
                        f1s.append(2*tp/(len(pred_s)+len(true_s)))
                m = 0.4*acc_d + 0.1*np.mean(f1d) + 0.4*acc_s + 0.1*np.mean(f1s)
                if m > best_m:
                    best_m = m
                    best_params = (td, ts, ks)
    return acc_d, acc_s, best_m, best_params


def main():
    print("加载 checkpoint...")
    ckpt = torch.load(CHECKPOINT_IN, map_location='cpu', weights_only=False)
    main_diag_classes = ckpt['main_diag_classes']
    main_surg_classes = ckpt['main_surg_classes']
    other_diag_list = ckpt['other_diag_list']
    other_surg_list = ckpt['other_surg_list']
    main_diag2idx = {c: i for i, c in enumerate(main_diag_classes)}
    main_surg2idx = {c: i for i, c in enumerate(main_surg_classes)}
    other_diag2idx = ckpt['other_diag2idx']
    other_surg2idx = ckpt['other_surg2idx']

    NUM_MAIN_DIAG = len(main_diag_classes)
    NUM_MAIN_SURG = len(main_surg_classes)
    NUM_OTHER_DIAG = len(other_diag_list)
    NUM_OTHER_SURG = len(other_surg_list)
    print(f"  主诊断 {NUM_MAIN_DIAG}, 主手术 {NUM_MAIN_SURG}, "
          f"其他诊断 {NUM_OTHER_DIAG}, 其他手术 {NUM_OTHER_SURG}")

    print("加载 LLM...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True)
    llm = AutoModel.from_pretrained(
        BASE_MODEL, device_map="cuda", trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float16)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    model = ICDClassifier(llm, embed_dim=EMBED_DIM,
                          num_main_diag=NUM_MAIN_DIAG, num_main_surg=NUM_MAIN_SURG,
                          num_other_diag=NUM_OTHER_DIAG, num_other_surg=NUM_OTHER_SURG).to(device)
    model.load_state_dict(ckpt['model_state'])
    print("  已从 best_classifier.pt 加载权重")

    print("加载数据...")
    train_data = json.load(open(TRAIN_FILE, encoding='utf-8'))
    dev_data = json.load(open(DEV_FILE, encoding='utf-8'))
    print(f"  train: {len(train_data)}, dev: {len(dev_data)}")

    train_ds = ICDDataset(train_data, tokenizer, main_diag2idx, main_surg2idx,
                          other_diag2idx, other_surg2idx, max_len=MAX_LEN)
    dev_ds = ICDDataset(dev_data, tokenizer, main_diag2idx, main_surg2idx,
                        other_diag2idx, other_surg2idx, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2)

    # 只训练 MLP/classifier 头，LLM 冻结
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    best_m = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            iids = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            ld = batch['label_d'].to(device)
            ls = batch['label_s'].to(device)
            od = batch['od'].to(device)
            os_ = batch['os'].to(device)

            pd, ps, p_od, p_os = model(iids, amask)
            loss = ce(pd, ld) + ce(ps, ls) + bce(p_od, od) + bce(p_os, os_)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 500 == 0:
                print(f"  step {step+1}/{len(train_loader)} loss={total_loss/(step+1):.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg_loss={avg_loss:.4f}")

        print("  评估 dev...")
        acc_d, acc_s, m, best_params = eval_model(
            model, dev_loader, main_diag_classes, main_surg_classes,
            other_diag2idx, other_surg2idx, dev_data)
        td, ts, ks = best_params
        print(f"  Acc_d={acc_d:.4f} Acc_s={acc_s:.4f} M={m:.4f} (td={td} ts={ts} ks={ks})")

        if m > best_m:
            best_m = m
            torch.save({
                'model_state': model.state_dict(),
                'main_diag_classes': main_diag_classes,
                'main_surg_classes': main_surg_classes,
                'other_diag_list': other_diag_list,
                'other_surg_list': other_surg_list,
                'other_diag2idx': other_diag2idx,
                'other_surg2idx': other_surg2idx,
                'best_td': td, 'best_ts': ts, 'best_ks': ks,
            }, CHECKPOINT_OUT)
            print(f"  => 保存 best_classifier_v2.pt (M={best_m:.4f})")

    print(f"\n训练完成，最优 M={best_m:.4f}")


if __name__ == "__main__":
    main()
