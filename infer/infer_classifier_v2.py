"""
用 best_classifier.pt 跑测试集 A_test.xlsx，用最优阈值 (td=0.25, ts=0.5) 生成 submission。
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
OUTPUT_FILE = "/root/autodl-tmp/predictions_classifier_v2.csv"
SUBMISSION_FILE = "/root/autodl-tmp/submission_classifier_v2.txt"

MAX_LEN = 512
BATCH_SIZE = 4
EMBED_DIM = 512
TD = 0.25  # other_diag 阈值
TS = 0.50  # other_surg 阈值
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
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
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
    print(f"  {len(test_df)} samples")

    test_ds = TestDataset(test_df, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"开始推理 ({len(test_loader)} batches)...")
    results = []
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ids = batch['id']
        with torch.no_grad():
            p_d, p_s, p_od, p_os = model(input_ids, attention_mask)
        probs_d = p_d.cpu().numpy()
        probs_s = p_s.cpu().numpy()
        od_probs = torch.sigmoid(p_od).cpu().numpy()
        os_probs = torch.sigmoid(p_os).cpu().numpy()

        for i in range(len(ids)):
            main_d = main_diag_classes[probs_d[i].argmax()]
            main_s = main_surg_classes[probs_s[i].argmax()]

            # 新阈值：td=0.25, ts=0.5
            other_d_codes = [other_diag_list[j] for j in range(len(od_probs[i]))
                            if od_probs[i][j] > TD]
            other_d = ';'.join(other_d_codes)

            # top-2 + 阈值
            top2_idx = os_probs[i].argsort()[-2:][::-1]
            other_s_codes = [other_surg_list[j] for j in top2_idx
                            if os_probs[i][j] > TS]
            other_s = ';'.join(other_s_codes[:2])

            results.append({
                '病案标识': ids[i],
                '主要诊断编码': main_d,
                '其他诊断编码': other_d,
                '主要手术编码': main_s,
                '其他手术编码': other_s,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"已保存: {OUTPUT_FILE}")

    # 生成 submission（4 字段，无病案标识）
    print("生成 submission...")
    lines = []
    for _, row in result_df.iterrows():
        main_d = str(row['主要诊断编码']).strip() if pd.notna(row['主要诊断编码']) else ''
        other_d = str(row['其他诊断编码']).strip() if pd.notna(row['其他诊断编码']) else ''
        main_s = str(row['主要手术编码']).strip() if pd.notna(row['主要手术编码']) else ''
        other_s = str(row['其他手术编码']).strip() if pd.notna(row['其他手术编码']) else ''
        lines.append(f"{main_d}|{other_d}|{main_s}|{other_s}")

    with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"已保存: {SUBMISSION_FILE}, 共 {len(lines)} 行")

    # 统计
    avg_other_d = sum(len(l.split('|')[1].split(';')) if l.split('|')[1] else 0 for l in lines) / len(lines)
    avg_other_s = sum(len(l.split('|')[3].split(';')) if l.split('|')[3] else 0 for l in lines) / len(lines)
    print(f"平均其他诊断/样本: {avg_other_d:.2f}")
    print(f"平均其他手术/样本: {avg_other_s:.2f}")
    print("\n前 5 行预览:")
    for ln in lines[:5]:
        print(ln)


if __name__ == "__main__":
    main()
