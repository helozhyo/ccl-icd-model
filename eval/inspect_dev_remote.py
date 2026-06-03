import json
import pprint
from collections import Counter

d = json.load(open("/root/autodl-tmp/icd_data/dev.json", encoding="utf-8"))
print(len(d), "samples in dev.json")
print("Keys:", list(d[0].keys()))
print()
print("--- First sample (text truncated) ---")
sample = d[0]
sample['text'] = sample['text'][:200] + '...'
pprint.pprint(sample)
print()
print("--- Code distribution ---")
md = Counter(s.get('main_diag', '') for s in d)
ms = Counter(s.get('main_surg', '') for s in d)
print("Main diag:", len(md), "unique, top 10:", md.most_common(10))
print("Main surg:", len(ms), "unique, top 10:", ms.most_common(10))

od = Counter()
os_ = Counter()
for s in d:
    for c in s.get('other_diag', []):
        od[c] += 1
    for c in s.get('other_surg', []):
        os_[c] += 1
print("Other diag unique:", len(od), "top 5:", od.most_common(5))
print("Other surg unique:", len(os_), "top 5:", os_.most_common(5))
print()
print("Avg other_diag per sample:", sum(len(s.get('other_diag', [])) for s in d) / len(d))
print("Avg other_surg per sample:", sum(len(s.get('other_surg', [])) for s in d) / len(d))
