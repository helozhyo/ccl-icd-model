#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCL26 ICD自动编码：冻结LLM + MLP分类头
第一步：只训练分类器，固定InternLM2-1.8B参数

主诊断(18类) + 主手术(16类)：softmax单标签分类
其他诊断 + 其他手术：多标签sigmoid分类
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc

# ========== 配置 ==========
BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
TRAIN_FILE = "/root/autodl-tmp/icd_data/train.json"
DEV_FILE = "/root/autodl-tmp/icd_data/dev.json"
CLASSES_FILE = "/root/autodl-tmp/icd_data/classes.json"
OUTPUT_DIR = "/root/autodl-tmp/classifier_output"

MAX_LEN = 512          # LLM输入最大长度
BATCH_SIZE = 2         # 3080Ti 12GB够用（冻结LLM，不需要存反向图）
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM = 512        # MLP中间层维度

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ========== 加载类别 ==========
with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    classes = json.load(f)

main_diag_classes = classes['main_diag_classes']   # 18个
main_surg_classes = classes['main_surg_classes']   # 16个

# 构建 label -> index 映射
main_diag2idx = {c: i for i, c in enumerate(main_diag_classes)}
main_surg2idx = {c: i for i, c in enumerate(main_surg_classes)}

NUM_MAIN_DIAG = len(main_diag_classes)
NUM_MAIN_SURG = len(main_surg_classes)

print(f"主诊断类别数: {NUM_MAIN_DIAG}")
print(f"主手术类别数: {NUM_MAIN_SURG}")
print(f"主诊断类别: {main_diag_classes}")
print(f"主手术类别: {main_surg_classes}")


# ========== 加载数据 ==========
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

train_data = load_json(TRAIN_FILE)
dev_data = load_json(DEV_FILE)

print(f"训练集: {len(train_data)} 条")
print(f"验证集: {len(dev_data)} 条")


# ========== 其他编码收集（用于多标签分类） ==========
# 收集训练集中出现过的所有其他诊断和手术编码
other_diag_set = set()
other_surg_set = set()

for item in train_data:
    other_d = item.get('other_diag', '')
    if other_d:
        for code in other_d.replace('；', ';').split(';'):
            code = code.strip()
            if code:
                other_diag_set.add(code)
    other_s = item.get('other_surg', '')
    if other_s:
        for code in other_s.replace('；', ';').split(';'):
            code = code.strip()
            if code:
                other_surg_set.add(code)

# DEV集也要算进来
for item in dev_data:
    other_d = item.get('other_diag', '')
    if other_d:
        for code in other_d.replace('；', ';').split(';'):
            code = code.strip()
            if code:
                other_diag_set.add(code)
    other_s = item.get('other_surg', '')
    if other_s:
        for code in other_s.replace('；', ';').split(';'):
            code = code.strip()
            if code:
                other_surg_set.add(code)

other_diag_list = sorted(other_diag_set)
other_surg_list = sorted(other_surg_set)
other_diag2idx = {c: i for i, c in enumerate(other_diag_list)}
other_surg2idx = {c: i for i, c in enumerate(other_surg_list)}

print(f"其他诊断唯一编码数: {len(other_diag_list)}")
print(f"其他手术唯一编码数: {len(other_surg_list)}")

NUM_OTHER_DIAG = len(other_diag_list)
NUM_OTHER_SURG = len(other_surg_list)


# ========== 数据集 ==========
class ICDDataset(Dataset):
    def __init__(self, data, main_diag2idx, main_surg2idx,
                 other_diag2idx, other_surg2idx, tokenizer, max_len=512):
        self.data = data
        self.main_diag2idx = main_diag2idx
        self.main_surg2idx = main_surg2idx
        self.other_diag2idx = other_diag2idx
        self.other_surg2idx = other_surg2idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text'][:1500]  # 截断到1500字符

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # 主诊断标签 (单标签)
        main_d = item.get('main_diag', '')
        main_d_label = self.main_diag2idx.get(main_d, -1)  # -1表示未知

        # 主手术标签 (单标签)
        main_s = item.get('main_surg', '')
        main_s_label = self.main_surg2idx.get(main_s, -1)

        # 其他诊断标签 (多标签)
        other_d_labels = torch.zeros(NUM_OTHER_DIAG)
        other_d = item.get('other_diag', '')
        if other_d:
            for code in other_d.replace('；', ';').split(';'):
                code = code.strip()
                if code in self.other_diag2idx:
                    other_d_labels[self.other_diag2idx[code]] = 1.0

        # 其他手术标签 (多标签)
        other_s_labels = torch.zeros(NUM_OTHER_SURG)
        other_s = item.get('other_surg', '')
        if other_s:
            for code in other_s.replace('；', ';').split(';'):
                code = code.strip()
                if code in self.other_surg2idx:
                    other_s_labels[self.other_surg2idx[code]] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'main_d_label': torch.tensor(main_d_label, dtype=torch.long),
            'main_s_label': torch.tensor(main_s_label, dtype=torch.long),
            'other_d_labels': other_d_labels,
            'other_s_labels': other_s_labels,
        }


# ========== 模型结构 ==========
class ICDClassifier(nn.Module):
    """
    冻结LLM，只训练分类头
    e_llm = mean_pool(LLM_output)
    e_fuse = e_llm (直接用LLM表征)
    p_main_d = softmax(MLP_dis(e_llm))   # 主诊断单标签
    p_main_s = softmax(MLP_dis(e_llm))   # 主手术单标签
    p_other_d = sigmoid(MLP_gen(e_llm))  # 其他诊断多标签
    p_other_s = sigmoid(MLP_gen(e_llm))   # 其他手术多标签
    """
    def __init__(self, llm, embed_dim=512):
        super().__init__()
        self.llm = llm
        self.hidden_dim = llm.config.hidden_size  # InternLM2 = 2048

        # 共享特征投影
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 主诊断分支 (18类单标签)
        self.mlp_dis_diag = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_main_diag = nn.Linear(embed_dim // 2, NUM_MAIN_DIAG)

        # 主手术分支 (16类单标签)
        self.mlp_dis_surg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_main_surg = nn.Linear(embed_dim // 2, NUM_MAIN_SURG)

        # 其他诊断分支 (多标签)
        self.mlp_gen_diag = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_other_diag = nn.Linear(embed_dim, NUM_OTHER_DIAG)

        # 其他手术分支 (多标签)
        self.mlp_gen_surg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cls_other_surg = nn.Linear(embed_dim, NUM_OTHER_SURG)

    def forward(self, input_ids, attention_mask):
        # LLM forward，冻结参数
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
            # hidden_states[-1] = 最后一层 hidden state
            hidden = outputs.hidden_states[-1]  # [B, L, hidden_dim]

        # Mean pooling (hidden_states[-1] 是 fp16，转 fp32 供投影层使用)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        e_llm = (hidden.float() * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # 特征投影
        e_fuse = self.proj(e_llm)

        # 四个分类头
        e_dis_d = self.mlp_dis_diag(e_fuse)
        p_main_d = self.cls_main_diag(e_dis_d)

        e_dis_s = self.mlp_dis_surg(e_fuse)
        p_main_s = self.cls_main_surg(e_dis_s)

        e_gen_d = self.mlp_gen_diag(e_fuse)
        p_other_d = self.cls_other_diag(e_gen_d)

        e_gen_s = self.mlp_gen_surg(e_fuse)
        p_other_s = self.cls_other_surg(e_gen_s)

        return p_main_d, p_main_s, p_other_d, p_other_s


# ========== 损失函数 ==========
# 主诊断/主手术：交叉熵，忽略-1标签（未知类别）
criterion_ce = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)

# 其他诊断/其他手术：二值交叉熵
criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')


def compute_loss(p_main_d, p_main_s, p_other_d, p_other_s,
                 main_d_labels, main_s_labels, other_d_labels, other_s_labels,
                 alpha_d=0.4, alpha_s=0.4, beta_d=0.1, beta_s=0.1):
    # 忽略未知标签的样本（标签=-1）
    valid_d = main_d_labels >= 0
    valid_s = main_s_labels >= 0

    L_main_d = criterion_ce(p_main_d[valid_d], main_d_labels[valid_d])
    L_main_s = criterion_ce(p_main_s[valid_s], main_s_labels[valid_s])
    L_other_d = criterion_bce(p_other_d, other_d_labels)
    L_other_s = criterion_bce(p_other_s, other_s_labels)

    L_total = alpha_d * L_main_d + alpha_s * L_main_s + beta_d * L_other_d + beta_s * L_other_s
    return L_total, L_main_d, L_main_s, L_other_d, L_other_s


# ========== 评测函数 ==========
def safe_f1(y_true_list, y_pred_list):
    """计算micro-F1，处理空列表"""
    valid = [(t, p) for t, p in zip(y_true_list, y_pred_list) if t or p]
    if not valid:
        return 0.0
    y_t = [set(x.split(';')) if x else set() for x, _ in valid]
    y_p = [set(x.split(';')) if x else set() for _, x in valid]
    all_labels = set()
    for s in y_t + y_p:
        all_labels.update(s)
    all_labels = sorted(all_labels)
    if not all_labels:
        return 0.0
    y_t_bin = [[1 if lab in s else 0 for lab in all_labels] for s in y_t]
    y_p_bin = [[1 if lab in s else 0 for lab in all_labels] for s in y_p]
    from sklearn.metrics import f1_score
    return f1_score(y_t_bin, y_p_bin, average='micro', zero_division=0)


def evaluate(model, dev_loader, main_diag_classes, main_surg_classes,
             other_diag_list, other_surg_list, other_diag2idx, other_surg2idx):
    model.eval()
    all_main_d_pred, all_main_d_true = [], []
    all_main_s_pred, all_main_s_true = [], []
    all_other_d_pred, all_other_d_true = [], []
    all_other_s_pred, all_other_s_true = [], []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            main_d_labels = batch['main_d_label'].numpy()
            main_s_labels = batch['main_s_label'].numpy()
            other_d_labels = batch['other_d_labels'].numpy()
            other_s_labels = batch['other_s_labels'].numpy()

            p_main_d, p_main_s, p_other_d, p_other_s = model(input_ids, attention_mask)

            # 主诊断
            for i, label in enumerate(main_d_labels):
                if label >= 0:
                    pred = main_diag_classes[p_main_d[i].argmax().item()]
                    all_main_d_pred.append(pred)
                    all_main_d_true.append(main_diag_classes[label])

            # 主手术
            for i, label in enumerate(main_s_labels):
                if label >= 0:
                    pred = main_surg_classes[p_main_s[i].argmax().item()]
                    all_main_s_pred.append(pred)
                    all_main_s_true.append(main_surg_classes[label])

            # 其他诊断
            probs_d = p_other_d.cpu().numpy()
            for i in range(len(probs_d)):
                codes = []
                for j, p in enumerate(probs_d[i]):
                    if p > 0.5:
                        codes.append(other_diag_list[j])
                all_other_d_pred.append(';'.join(codes))
                true_codes = [other_diag_list[j] for j in range(len(other_d_labels[i])) if other_d_labels[i][j] > 0.5]
                all_other_d_true.append(';'.join(true_codes))

            # 其他手术（最多2个）
            probs_s = p_other_s.cpu().numpy()
            for i in range(len(probs_s)):
                top2_idx = probs_s[i].argsort()[-2:][::-1]
                codes = [other_surg_list[j] for j in top2_idx if probs_s[i][j] > 0.3]
                codes = codes[:2]
                all_other_s_pred.append(';'.join(codes))
                true_codes = [other_surg_list[j] for j in range(len(other_s_labels[i])) if other_s_labels[i][j] > 0.5]
                all_other_s_true.append(';'.join(true_codes))

    # 计算指标
    acc_main_d = sum(1 for p, t in zip(all_main_d_pred, all_main_d_true) if p == t) / len(all_main_d_true)
    acc_main_s = sum(1 for p, t in zip(all_main_s_pred, all_main_s_true) if p == t) / len(all_main_s_true)
    f1_other_d = safe_f1(all_other_d_true, all_other_d_pred)
    f1_other_s = safe_f1(all_other_s_true, all_other_s_pred)

    M_total = 0.4 * acc_main_d + 0.1 * f1_other_d + 0.4 * acc_main_s + 0.1 * f1_other_s

    return {
        'Acc_main_d': acc_main_d,
        'Acc_main_s': acc_main_s,
        'F1_other_d': f1_other_d,
        'F1_other_s': f1_other_s,
        'M_total': M_total,
        'detail': f"Acc_main_d={acc_main_d:.4f}, Acc_main_s={acc_main_s:.4f}, "
                  f"F1_other_d={f1_other_d:.4f}, F1_other_s={f1_other_s:.4f}, M_total={M_total:.4f}"
    }


# ========== 训练 ==========
def train():
    print("\n=== 加载模型 ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    llm = AutoModel.from_pretrained(BASE_MODEL, device_map="cuda",
                                    trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16)
    llm.eval()
    # 冻结LLM
    for param in llm.parameters():
        param.requires_grad = False

    print(f"LLM参数已冻结，隐藏维度: {llm.config.hidden_size}")

    model = ICDClassifier(llm, embed_dim=EMBED_DIM).to(device)
    print(f"模型参数量（仅分类头）: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 只优化分类头参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 数据集
    train_dataset = ICDDataset(train_data, main_diag2idx, main_surg2idx,
                              other_diag2idx, other_surg2idx, tokenizer, max_len=MAX_LEN)
    dev_dataset = ICDDataset(dev_data, main_diag2idx, main_surg2idx,
                             other_diag2idx, other_surg2idx, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    print(f"\n训练集: {len(train_dataset)}, 批次数: {len(train_loader)}")
    print(f"验证集: {len(dev_dataset)}, 批次数: {len(dev_loader)}")

    best_m_total = 0.0
    best_results = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            main_d_labels = batch['main_d_label'].to(device)
            main_s_labels = batch['main_s_label'].to(device)
            other_d_labels = batch['other_d_labels'].to(device)
            other_s_labels = batch['other_s_labels'].to(device)

            optimizer.zero_grad()
            p_main_d, p_main_s, p_other_d, p_other_s = model(input_ids, attention_mask)
            loss, L_md, L_ms, L_od, L_os = compute_loss(
                p_main_d, p_main_s, p_other_d, p_other_s,
                main_d_labels, main_s_labels, other_d_labels, other_s_labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # 评测
        results = evaluate(model, dev_loader, main_diag_classes, main_surg_classes,
                         other_diag_list, other_surg_list, other_diag2idx, other_surg2idx)
        print(f"评测结果: {results['detail']}")

        # 保存最优模型
        if results['M_total'] > best_m_total:
            best_m_total = results['M_total']
            best_results = results
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'results': results,
                'main_diag_classes': main_diag_classes,
                'main_surg_classes': main_surg_classes,
                'other_diag_list': other_diag_list,
                'other_surg_list': other_surg_list,
                'other_diag2idx': other_diag2idx,
                'other_surg2idx': other_surg2idx,
            }, os.path.join(OUTPUT_DIR, 'best_classifier.pt'))
            print(f"  >>> 已保存最优模型 (M_total={best_m_total:.4f})")

        # 每个epoch后清理显存
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n训练完成！最优 M_total={best_m_total:.4f}")
    print(f"最优结果: {best_results['detail']}")
    return model, best_results


if __name__ == "__main__":
    train()
