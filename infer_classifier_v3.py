"""
v3: 试 3 种不同参数组合，生成 3 个 submission 看哪个分数高
1. v3a: ts=0.4 + top-3（更多 top-k 召回）
2. v3b: ts=0.35 + top-2（中等降阈）
3. v3c: td=0.2 + ts=0.45（再降一点 td）
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd

BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
CHECKPOINT = "/root/autodl-tmp/classifier_output/best_classifier.pt"

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


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def concat_text(self, row):
        parts = []
        for col_idx in range(1, 15):
            val = row.iloc[col_idx] if col_idx < len(row) else None
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return '\n'.join(parts)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_ = str(row.iloc[0]) if pd.notna(row.iloc[0]) else str(idx)
        text = self.concat_text(row)
        text = text[:1500]
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True,
            max_length=self.max_len, padding='max_length')
        return {
            'id': id_,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }


def main():
    print("加载checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    main_diag_classes = ckpt['main_diag_classes']
    main_surg_classes = ckpt['main_surg_classes']
    other_diag_list = ckpt['other_diag_list']
    other_surg_list = ckpt['other_surg_list']

    NUM_MAIN_DIAG = len(main_diag_classes)
    NUM_MAIN_SURG = len(main_surg_classes)
    NUM_OTHER_DIAG = len(other_diag_list)
    NUM_OTHER_SURG = len(other_surg_list)
    print(f"主诊断 {NUM_MAIN_DIAG}, 主手术 {NUM_MAIN_SURG}, 其他诊断 {NUM_OTHER_DIAG}, 其他手术 {NUM_OTHER_SURG}")

    print("加载LLM...")
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

    print("加载测试集...")
    test_df = pd.read_excel(TEST_FILE, engine='openpyxl')

    test_ds = TestDataset(test_df, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"开始推理 ({len(test_loader)} batches)...")
    # 收集所有 logits/probs
    all_ids = []
    all_p_d = []
    all_p_s = []
    all_od = []
    all_os = []
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ids = batch['id']
        with torch.no_grad():
            p_d, p_s, p_od, p_os = model(input_ids, attention_mask)
        all_ids.extend(ids)
        all_p_d.append(p_d.cpu())
        all_p_s.append(p_s.cpu())
        all_od.append(p_od.cpu())
        all_os.append(p_os.cpu())

    p_d = torch.cat(all_p_d).numpy()
    p_s = torch.cat(all_p_s).numpy()
    od = torch.sigmoid(torch.cat(all_od)).numpy()
    os_ = torch.sigmoid(torch.cat(all_os)).numpy()

    # 写一个函数：用给定阈值生成 submission
    def gen_submission(td, ts, k_s, filename):
        results = []
        for i in range(len(all_ids)):
            main_d = main_diag_classes[p_d[i].argmax()]
            main_s = main_surg_classes[p_s[i].argmax()]
            other_d_codes = [other_diag_list[j] for j in range(len(od[i]))
                            if od[i][j] > td]
            other_d = ';'.join(other_d_codes)
            if k_s is None:
                other_s_codes = [other_surg_list[j] for j in range(len(os_[i]))
                                if os_[i][j] > ts]
            else:
                top_idx = os_[i].argsort()[-k_s:][::-1]
                other_s_codes = [other_surg_list[j] for j in top_idx
                                if os_[i][j] > ts]
            other_s = ';'.join(other_s_codes[:k_s] if k_s else other_s_codes)
            results.append(f"{main_d}|{other_d}|{main_s}|{other_s}")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results) + '\n')
        # 统计
        n_d = [len(r.split('|')[1].split(';')) if r.split('|')[1] else 0 for r in results]
        n_s = [len(r.split('|')[3].split(';')) if r.split('|')[3] else 0 for r in results]
        print(f"  {filename}: avg_other_d={sum(n_d)/len(n_d):.2f}, avg_other_s={sum(n_s)/len(n_s):.2f}")

    # 4 个变体
    gen_submission(0.25, 0.50, 2, '/root/autodl-tmp/sub_v3a.txt')  # 与 v2 一致（基线）
    gen_submission(0.25, 0.40, 3, '/root/autodl-tmp/sub_v3b.txt')  # 降 ts + 放大 top-k
    gen_submission(0.20, 0.45, 3, '/root/autodl-tmp/sub_v3c.txt')  # 全降
    gen_submission(0.25, 0.45, 4, '/root/autodl-tmp/sub_v3d.txt')  # 保留 td, 降 ts, 放大 k


if __name__ == "__main__":
    main()
