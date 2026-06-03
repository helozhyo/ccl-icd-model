import json
from collections import Counter

d = json.load(open("/root/autodl-tmp/icd_data/dev.json", encoding="utf-8"))

od = Counter()
os_ = Counter()
od_per = []
os_per = []
for s in d:
    ods = [c.strip() for c in s['other_diag'].replace('；', ';').split(';') if c.strip()]
    oss = [c.strip() for c in s['other_surg'].replace('；', ';').split(';') if c.strip()]
    od_per.append(len(ods))
    os_per.append(len(oss))
    for c in ods: od[c] += 1
    for c in oss: os_[c] += 1

print(f"Dev 180 samples")
print(f"  other_diag unique codes: {len(od)}")
print(f"  other_surg unique codes: {len(os_)}")
print(f"  avg other_diag per sample: {sum(od_per)/len(od_per):.2f}, max: {max(od_per)}, min: {min(od_per)}")
print(f"  avg other_surg per sample: {sum(os_per)/len(os_per):.2f}, max: {max(os_per)}, min: {min(os_per)}")
print()
print(f"Top 30 other_diag codes:")
for k, v in od.most_common(30):
    print(f"  {k}: {v}")
print()
print(f"Top 30 other_surg codes:")
for k, v in os_.most_common(30):
    print(f"  {k}: {v}")

# 跟 train.json 比
print()
print("=== Train.json ===")
t = json.load(open("/root/autodl-tmp/icd_data/train.json", encoding="utf-8"))
print(f"  Train samples: {len(t)}")
od_t, os_t = Counter(), Counter()
for s in t:
    ods = [c.strip() for c in s['other_diag'].replace('；', ';').split(';') if c.strip()]
    oss = [c.strip() for c in s['other_surg'].replace('；', ';').split(';') if c.strip()]
    for c in ods: od_t[c] += 1
    for c in oss: os_t[c] += 1
print(f"  other_diag unique: {len(od_t)}")
print(f"  other_surg unique: {len(os_t)}")
