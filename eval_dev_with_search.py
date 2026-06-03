"""
对 best_classifier.pt 在 dev 集上重新跑一次推理，存 raw logits。
然后做阈值网格搜索找最优 (td, ts)。
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ========== 配置 ==========
BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
DEV_FILE = "/root/autodl-tmp/icd_data/dev.json"
CHECKPOINT = "/root/autodl-tmp/classifier_output/best_classifier.pt"
LOGITS_OUT = "/root/autodl-tmp/dev_logits.npz"
PROBS_OUT = "/root/autodl-tmp/dev_probs.npz"

MAX_LEN = 512
BATCH_SIZE = 4
EMBED_DIM = 512
device = torch.device("cuda")


class ICDClassifier(nn.Module):
    def __init__(self, llm, embed_dim=512, num_main_diag=18, num_main_surg=16,
                 num_other_diag=1, num_other_surg=1):
        super().__init__()
        self.llm = llm
        self.hidden_dim = llm.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
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


class DevDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')[:1500]
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding='max_length'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }


def parse_codes(s):
    if not s:
        return []
    return [c.strip() for c in s.replace('；', ';').split(';') if c.strip()]


def main():
    print("=== 加载 checkpoint ===")
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    main_diag_classes = ckpt['main_diag_classes']
    main_surg_classes = ckpt['main_surg_classes']
    other_diag_list = ckpt['other_diag_list']
    other_surg_list = ckpt['other_surg_list']
    other_diag2idx = ckpt['other_diag2idx']
    other_surg2idx = ckpt['other_surg2idx']

    NUM_MAIN_DIAG = len(main_diag_classes)
    NUM_MAIN_SURG = len(main_surg_classes)
    NUM_OTHER_DIAG = len(other_diag_list)
    NUM_OTHER_SURG = len(other_surg_list)

    print(f"  Main diag: {NUM_MAIN_DIAG}, Main surg: {NUM_MAIN_SURG}")
    print(f"  Other diag: {NUM_OTHER_DIAG}, Other surg: {NUM_OTHER_SURG}")

    print("\n=== 加载 LLM ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    llm = AutoModel.from_pretrained(BASE_MODEL, device_map="cuda",
                                    trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    model = ICDClassifier(llm, embed_dim=EMBED_DIM,
                          num_main_diag=NUM_MAIN_DIAG, num_main_surg=NUM_MAIN_SURG,
                          num_other_diag=NUM_OTHER_DIAG, num_other_surg=NUM_OTHER_SURG).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print("\n=== 加载 dev 集 ===")
    dev_data = json.load(open(DEV_FILE, encoding="utf-8"))
    print(f"  {len(dev_data)} samples")

    dev_ds = DevDataset(dev_data, tokenizer, max_len=MAX_LEN)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("\n=== 跑推理 ===")
    all_p_main_d = []
    all_p_main_s = []
    all_p_other_d = []  # logits
    all_p_other_s = []  # logits
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            p_d, p_s, p_od, p_os = model(input_ids, attention_mask)
            all_p_main_d.append(p_d.cpu())
            all_p_main_s.append(p_s.cpu())
            all_p_other_d.append(p_od.cpu())
            all_p_other_s.append(p_os.cpu())

    p_main_d = torch.cat(all_p_main_d).numpy()
    p_main_s = torch.cat(all_p_main_s).numpy()
    logits_other_d = torch.cat(all_p_other_d).numpy()
    logits_other_s = torch.cat(all_p_other_s).numpy()
    probs_other_d = 1 / (1 + np.exp(-logits_other_d))  # sigmoid
    probs_other_s = 1 / (1 + np.exp(-logits_other_s))

    # 准备 ground truth
    gt_main_d = np.array([d['main_diag'] for d in dev_data])
    gt_main_s = np.array([d['main_surg'] for d in dev_data])
    gt_other_d = np.zeros((len(dev_data), NUM_OTHER_DIAG), dtype=np.float32)
    gt_other_s = np.zeros((len(dev_data), NUM_OTHER_SURG), dtype=np.float32)
    for i, d in enumerate(dev_data):
        for c in parse_codes(d.get('other_diag', '')):
            if c in other_diag2idx:
                gt_other_d[i, other_diag2idx[c]] = 1.0
        for c in parse_codes(d.get('other_surg', '')):
            if c in other_surg2idx:
                gt_other_s[i, other_surg2idx[c]] = 1.0

    # 主诊断/主手术准确率
    pred_main_d = np.array([main_diag_classes[p.argmax()] for p in p_main_d])
    pred_main_s = np.array([main_surg_classes[p.argmax()] for p in p_main_s])
    acc_d = (pred_main_d == gt_main_d).mean()
    acc_s = (pred_main_s == gt_main_s).mean()
    print(f"\n=== 硬指标 ===")
    print(f"Acc_main_d: {acc_d:.4f}")
    print(f"Acc_main_s: {acc_s:.4f}")

    # 搜索最优阈值
    print("\n=== 阈值搜索 ===")
    best = None
    best_m = 0
    for td in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for ts in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            # 计算 F1
            f1_d_list = []
            for i in range(len(dev_data)):
                pred = set(np.where(probs_other_d[i] > td)[0])
                true = set(np.where(gt_other_d[i] > 0)[0])
                if not pred and not true:
                    f1_d_list.append(1.0)
                elif not pred or not true:
                    f1_d_list.append(0.0)
                else:
                    p = len(pred & true) / len(pred)
                    r = len(pred & true) / len(true)
                    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
                    f1_d_list.append(f1)
            f1_d = np.mean(f1_d_list)

            f1_s_list = []
            for i in range(len(dev_data)):
                pred = set(np.where(probs_other_s[i] > ts)[0])
                true = set(np.where(gt_other_s[i] > 0)[0])
                if not pred and not true:
                    f1_s_list.append(1.0)
                elif not pred or not true:
                    f1_s_list.append(0.0)
                else:
                    p = len(pred & true) / len(pred)
                    r = len(pred & true) / len(true)
                    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
                    f1_s_list.append(f1)
            f1_s = np.mean(f1_s_list)

            m_total = 0.4*acc_d + 0.1*f1_d + 0.4*acc_s + 0.1*f1_s
            if m_total > best_m:
                best_m = m_total
                best = (td, ts, f1_d, f1_s, m_total)
            print(f"  td={td:.2f} ts={ts:.2f} | F1_d={f1_d:.4f} F1_s={f1_s:.4f} M_total={m_total:.4f}")

    print(f"\n=== 最优阈值: td={best[0]:.2f}, ts={best[1]:.2f} ===")
    print(f"  F1_d={best[2]:.4f} F1_s={best[3]:.4f} M_total={best[4]:.4f}")

    # 保存 raw probs
    np.savez(PROBS_OUT,
             p_main_d=p_main_d, p_main_s=p_main_s,
             probs_other_d=probs_other_d, probs_other_s=probs_other_s,
             gt_main_d=gt_main_d, gt_main_s=gt_main_s,
             gt_other_d=gt_other_d, gt_other_s=gt_other_s)
    print(f"\n保存到 {PROBS_OUT}")


if __name__ == "__main__":
    main()
