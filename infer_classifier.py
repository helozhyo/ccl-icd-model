#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的分类器对测试集推理，生成提交文件
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd

# ========== 配置 ==========
BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
CHECKPOINT = "/root/autodl-tmp/classifier_output/best_classifier.pt"
OUTPUT_FILE = "/root/autodl-tmp/predictions_classifier.csv"
SUBMISSION_FILE = "/root/autodl-tmp/submission_classifier.txt"

MAX_LEN = 512
BATCH_SIZE = 4
EMBED_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== 模型结构（与训练时一致） ==========
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
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_main_diag = nn.Linear(embed_dim // 2, num_main_diag)

        self.mlp_dis_surg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_main_surg = nn.Linear(embed_dim // 2, num_main_surg)

        self.mlp_gen_diag = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_other_diag = nn.Linear(embed_dim, num_other_diag)

        self.mlp_gen_surg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
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


# ========== 测试数据集 ==========
class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def concat_text(self, row):
        """与训练时一致的文本拼接方式：取前14列（跳过第一列ID）"""
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
        text = text[:1500]  # 截断

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding='max_length'
        )
        return {
            'id': id_,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }


def main():
    print("=" * 60)
    print("加载分类器...")
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')

    main_diag_classes = checkpoint['main_diag_classes']
    main_surg_classes = checkpoint['main_surg_classes']
    other_diag_list = checkpoint['other_diag_list']
    other_surg_list = checkpoint['other_surg_list']
    other_diag2idx = checkpoint['other_diag2idx']
    other_surg2idx = checkpoint['other_surg2idx']

    NUM_MAIN_DIAG = len(main_diag_classes)
    NUM_MAIN_SURG = len(main_surg_classes)
    NUM_OTHER_DIAG = len(other_diag_list)
    NUM_OTHER_SURG = len(other_surg_list)

    print(f"主诊断类别数: {NUM_MAIN_DIAG}")
    print(f"主手术类别数: {NUM_MAIN_SURG}")
    print(f"其他诊断类别数: {NUM_OTHER_DIAG}")
    print(f"其他手术类别数: {NUM_OTHER_SURG}")

    print("\n加载LLM...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    llm = AutoModel.from_pretrained(BASE_MODEL, device_map="cuda",
                                    trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    model = ICDClassifier(llm, embed_dim=EMBED_DIM,
                          num_main_diag=NUM_MAIN_DIAG, num_main_surg=NUM_MAIN_SURG,
                          num_other_diag=NUM_OTHER_DIAG, num_other_surg=NUM_OTHER_SURG).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("模型加载完成")

    print("\n加载测试集...")
    test_df = pd.read_excel(TEST_FILE, engine='openpyxl')
    print(f"测试集样本数: {len(test_df)}")

    test_dataset = TestDataset(test_df, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\n开始推理 ({len(test_loader)} batches)...")

    results = []
    for batch in tqdm(test_loader, desc="推理"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ids = batch['id']

        with torch.no_grad():
            p_main_d, p_main_s, p_other_d, p_other_s = model(input_ids, attention_mask)

        probs_d = p_main_d.cpu().numpy()
        probs_s = p_main_s.cpu().numpy()
        other_d_probs = torch.sigmoid(p_other_d).cpu().numpy()
        other_s_probs = torch.sigmoid(p_other_s).cpu().numpy()

        for i in range(len(ids)):
            # 主诊断：取概率最高的1个
            main_d = main_diag_classes[probs_d[i].argmax()]

            # 主手术：取概率最高的1个
            main_s = main_surg_classes[probs_s[i].argmax()]

            # 其他诊断：所有 > 0.35 的编码（dev最优阈值）
            other_d_codes = [other_diag_list[j] for j in range(len(other_d_probs[i]))
                            if other_d_probs[i][j] > 0.35]
            other_d = ';'.join(other_d_codes)

            # 其他手术：概率最高的2个（最多2个，阈值0.5）
            top2_idx = other_s_probs[i].argsort()[-2:][::-1]
            other_s_codes = [other_surg_list[j] for j in top2_idx
                            if other_s_probs[i][j] > 0.5]
            other_s = ';'.join(other_s_codes[:2])

            results.append({
                '病案标识': ids[i],
                '主要诊断编码': main_d,
                '其他诊断编码': other_d,
                '主要手术编码': main_s,
                '其他手术编码': other_s,
            })

    # 保存CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存: {OUTPUT_FILE}")

    # 生成提交文件
    print("\n生成提交文件...")
    lines = []
    for _, row in result_df.iterrows():
        id_ = str(row['病案标识']).strip()
        main_d = str(row['主要诊断编码']).strip() if pd.notna(row['主要诊断编码']) else ''
        other_d = str(row['其他诊断编码']).strip() if pd.notna(row['其他诊断编码']) else ''
        main_s = str(row['主要手术编码']).strip() if pd.notna(row['主要手术编码']) else ''
        other_s = str(row['其他手术编码']).strip() if pd.notna(row['其他手术编码']) else ''
        lines.append(f"{id_}\t{main_d}|{other_d}|{main_s}|{other_s}")

    with open(SUBMISSION_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"提交文件已保存: {SUBMISSION_FILE}")
    print(f"共 {len(lines)} 条")

    # 预览
    print("\n前10行预览:")
    for line in lines[:10]:
        print(line)
    print("\n后5行:")
    for line in lines[-5:]:
        print(line)

    # 统计
    main_d_dist = result_df['主要诊断编码'].value_counts()
    print(f"\n主诊断分布(前10):")
    for code, n in main_d_dist.head(10).items():
        print(f"  {code}: {n}")
    print(f"主诊断唯一值数: {main_d_dist.nunique()}")


if __name__ == "__main__":
    main()
