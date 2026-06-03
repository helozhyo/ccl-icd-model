import json
d = json.load(open("/root/autodl-tmp/icd_data/dev.json", encoding="utf-8"))

# 详细看 other_diag / other_surg 的样子
for i in range(5):
    s = d[i]
    print(f"=== sample {i} ===")
    print(f"  main_diag: {repr(s['main_diag'])}")
    print(f"  main_surg: {repr(s['main_surg'])}")
    print(f"  other_diag: type={type(s['other_diag']).__name__}, value={repr(s['other_diag'])[:200]}")
    print(f"  other_surg: type={type(s['other_surg']).__name__}, value={repr(s['other_surg'])[:200]}")
    print(f"  output: {repr(s['output'])}")
    print()

# 看output 字段
print("--- output 字段内容（看格式）---")
for i in range(3):
    print(d[i]['output'])
