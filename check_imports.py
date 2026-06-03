#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/miniconda3/lib/python3.10/site-packages')

import torch
print(f"torch={torch.__version__}, amp={'yes' if hasattr(torch.cuda, 'amp') else 'no'}")
print(f"autocast={'yes' if hasattr(torch.cuda.amp, 'autocast') else 'no'}")
print(f"GradScaler={'yes' if hasattr(torch.cuda.amp, 'GradScaler') else 'no'}")

try:
    from transformers import get_cosine_schedule_with_warmup
    print("get_cosine_schedule_with_warmup from transformers: OK")
except:
    print("get_cosine_schedule_with_warmup from transformers: FAIL")
    # try torch
    try:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        print("CosineAnnealingWarmRestarts from torch: OK")
    except:
        print("No cosine scheduler available")

try:
    from torch.utils.data import Dataset, DataLoader
    print("DataLoader: OK")
except Exception as e:
    print(f"DataLoader: FAIL {e}")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("peft: OK")
except Exception as e:
    print(f"peft: FAIL {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("transformers models: OK")
except Exception as e:
    print(f"transformers models: FAIL {e}")

try:
    from tqdm import tqdm
    print("tqdm: OK")
except Exception as e:
    print(f"tqdm: FAIL {e}")
