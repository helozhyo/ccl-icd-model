import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

BASE_MODEL = "/root/autodl-tmp/models/internlm_InternLM2-1_8B"
TEST_FILE = "/root/autodl-tmp/A_test.xlsx"
CHECKPOINT = "/root/autodl-tmp/classifier_output/best_classifier_v3.pt"
MAX_LEN, BATCH_SIZE, EMBED_DIM = 512, 4, 512
device = torch.device("cuda")


class ICDClassifier(nn.Module):
    def __init__(self, llm, embed_dim=512, num_main_diag=18, num_main_surg=16,
                 num_other_diag=1, num_other_surg=1):
        super().__init__()
        self.llm = llm
        self.hidden_dim = llm.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.mlp_dis_diag = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1))
        self.cls_main_diag = nn.Linear(embed_dim // 2, num_main_diag)
        self.mlp_dis_surg = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1))
        self.cls_main_surg = nn.Linear(embed_dim // 2, num_main_surg)
        self.mlp_gen_diag = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.cls_other_diag = nn.Linear(embed_dim, num_other_diag)
        self.mlp_gen_surg = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.1))
        self.cls_other_surg = nn.Linear(embed_dim, num_other_surg)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        e_llm = (hidden.float() * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        e_fuse = self.proj(e_llm)
        return (self.cls_main_diag(self.mlp_dis_diag(e_fuse)),
                self.cls_main_surg(self.mlp_dis_surg(e_fuse)),
                self.cls_other_diag(self.mlp_gen_diag(e_fuse)),
                self.cls_other_surg(self.mlp_gen_surg(e_fuse)))


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
        text = self.concat_text(row)[:1500]
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=self.max_len, padding='max_length')
        return {'id': id_,
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)}


def main():
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    main_diag_classes = ckpt['main_diag_classes']
    main_surg_classes = ckpt['main_surg_classes']
    other_diag_list = ckpt['other_diag_list']
    other_surg_list = ckpt['other_surg_list']
    td = ckpt.get('best_td', 0.25)
    ts = ckpt.get('best_ts', 0.40)
    ks = ckpt.get('best_ks', 2)
    print(f"Checkpoint best params: td={td} ts={ts} ks={ks}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    llm = AutoModel.from_pretrained(BASE_MODEL, device_map='cuda', trust_remote_code=True,
                                    local_files_only=True, torch_dtype=torch.float16)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    model = ICDClassifier(llm, embed_dim=EMBED_DIM,
                          num_main_diag=len(main_diag_classes), num_main_surg=len(main_surg_classes),
                          num_other_diag=len(other_diag_list), num_other_surg=len(other_surg_list)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    test_df = pd.read_excel(TEST_FILE, engine='openpyxl')
    test_ds = TestDataset(test_df, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_ids, all_pd, all_ps, all_od, all_os = [], [], [], [], []
    for batch in tqdm(test_loader):
        iids = batch['input_ids'].to(device)
        amask = batch['attention_mask'].to(device)
        with torch.no_grad():
            pd_, ps_, od_, os__ = model(iids, amask)
        all_ids.extend(batch['id'])
        all_pd.append(pd_.cpu()); all_ps.append(ps_.cpu())
        all_od.append(torch.sigmoid(od_).cpu())
        all_os.append(torch.sigmoid(os__).cpu())

    p_d = torch.cat(all_pd).numpy()
    p_s = torch.cat(all_ps).numpy()
    od = torch.cat(all_od).numpy()
    os_ = torch.cat(all_os).numpy()

    def gen(td_, ts_, ks_, fname):
        lines = []
        for i in range(len(all_ids)):
            main_d = main_diag_classes[p_d[i].argmax()]
            main_s = main_surg_classes[p_s[i].argmax()]
            other_d = ';'.join(other_diag_list[j] for j in range(len(od[i])) if od[i][j] > td_)
            top = os_[i].argsort()[-ks_:][::-1]
            other_s = ';'.join(other_surg_list[j] for j in top if os_[i][j] > ts_)
            lines.append(f'{main_d}|{other_d}|{main_s}|{other_s}')
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        nd = [len(l.split('|')[1].split(';')) if l.split('|')[1] else 0 for l in lines]
        ns = [len(l.split('|')[3].split(';')) if l.split('|')[3] else 0 for l in lines]
        print(f'{fname}: avg_d={sum(nd)/len(nd):.2f} avg_s={sum(ns)/len(ns):.2f}')

    # dev best
    gen(td, ts, ks, '/root/autodl-tmp/sub_v5a.txt')
    # 多几个变体
    gen(0.25, 0.40, 3, '/root/autodl-tmp/sub_v5b.txt')
    gen(0.20, 0.35, 3, '/root/autodl-tmp/sub_v5c.txt')
    gen(0.30, 0.40, 2, '/root/autodl-tmp/sub_v5d.txt')
    print('Done')


if __name__ == '__main__':
    main()
