"""
加载保存的 dev_probs.npz，做细粒度搜索。
支持 ts 0.1-0.5 步长 0.05，top-k 1-3。
"""
import numpy as np

data = np.load("/root/autodl-tmp/dev_probs.npz", allow_pickle=True)
p_main_d = data['p_main_d']
p_main_s = data['p_main_s']
probs_other_d = data['probs_other_d']
probs_other_s = data['probs_other_s']
gt_main_d = data['gt_main_d']
gt_main_s = data['gt_main_s']
gt_other_d = data['gt_other_d']
gt_other_s = data['gt_other_s']
main_diag_classes = data['gt_main_d']  # placeholder
main_surg_classes = data['gt_main_s']  # placeholder

# 重新载入 class 列表
import torch
ckpt = torch.load("/root/autodl-tmp/classifier_output/best_classifier.pt", map_location='cpu', weights_only=False)
main_diag_classes = ckpt['main_diag_classes']
main_surg_classes = ckpt['main_surg_classes']

# 主诊断/主手术准确率
pred_main_d = np.array([main_diag_classes[p.argmax()] for p in p_main_d])
pred_main_s = np.array([main_surg_classes[p.argmax()] for p in p_main_s])
acc_d = (pred_main_d == gt_main_d).mean()
acc_s = (pred_main_s == gt_main_s).mean()
print(f"Acc_main_d: {acc_d:.4f}, Acc_main_s: {acc_s:.4f}")

# F1 计算
def f1_macro(pred_sets, true_sets):
    """pred_sets, true_sets: list of sets, length N"""
    f1s = []
    for p, t in zip(pred_sets, true_sets):
        if not p and not t:
            f1s.append(1.0)
        elif not p or not t:
            f1s.append(0.0)
        else:
            tp = len(p & t)
            pr = tp / len(p)
            rc = tp / len(t)
            f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
            f1s.append(f1)
    return np.mean(f1s)


# 把 gt 变成 set 形式
gt_d_sets = [set(np.where(g > 0)[0]) for g in gt_other_d]
gt_s_sets = [set(np.where(g > 0)[0]) for g in gt_other_s]

# 详细搜索
print("\n=== 详细搜索 ===")
best = (0, 0, 0, 0, 0, 0, 0)
for td in [0.15, 0.20, 0.25, 0.30, 0.35]:
    for ts in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for kd in [None]:  # 其他诊断不限制 top-k，用阈值
            for ks in [2, 3, 4, 5, None]:  # 其他手术 top-k 限制
                pred_d = []
                for i in range(len(probs_other_d)):
                    if kd is None:
                        sel = set(np.where(probs_other_d[i] > td)[0])
                    else:
                        top = probs_other_d[i].argsort()[-kd:][::-1]
                        sel = set(top[probs_other_d[i][top] > td])
                    pred_d.append(sel)
                pred_s = []
                for i in range(len(probs_other_s)):
                    if ks is None:
                        sel = set(np.where(probs_other_s[i] > ts)[0])
                    else:
                        top = probs_other_s[i].argsort()[-ks:][::-1]
                        sel = set(top[probs_other_s[i][top] > ts])
                    pred_s.append(sel)
                f1_d = f1_macro(pred_d, gt_d_sets)
                f1_s = f1_macro(pred_s, gt_s_sets)
                m = 0.4*acc_d + 0.1*f1_d + 0.4*acc_s + 0.1*f1_s
                if m > best[0]:
                    best = (m, td, ts, kd, ks, f1_d, f1_s)
                    print(f"  NEW BEST: td={td} ts={ts} ks={ks} | F1_d={f1_d:.4f} F1_s={f1_s:.4f} M={m:.4f}")

print(f"\n=== 全局最优 ===")
m, td, ts, kd, ks, f1d, f1s = best
print(f"M_total={m:.4f}, td={td}, ts={ts}, kd={kd}, ks={ks}")
print(f"F1_d={f1d:.4f}, F1_s={f1s:.4f}")
