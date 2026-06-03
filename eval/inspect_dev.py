import sys; sys.path.insert(0, r'C:\Users\Hzh\Desktop\ccl-model-train')
from ssh_tool import ssh_exec

script = r'''
import json
data = json.load(open("/root/autodl-tmp/icd_data/dev.json", encoding="gbk"))
print(len(data), "samples")
print("Keys:", list(data[0].keys()))
import pprint
pprint.pprint(data[0])
'''

out, err = ssh_exec(f'python3 -c {repr(script)}', timeout=30)
print(out)
