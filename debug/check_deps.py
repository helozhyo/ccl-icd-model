#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/miniconda3/lib/python3.10/site-packages')

try:
    import peft
    print(f"peft={peft.__version__}")
except ImportError as e:
    print(f"peft NOT FOUND: {e}")

try:
    import deepspeed
    print(f"deepspeed={deepspeed.__version__}")
except ImportError as e:
    print(f"deepspeed NOT FOUND: {e}")

try:
    import accelerate
    print(f"accelerate={accelerate.__version__}")
except ImportError as e:
    print(f"accelerate NOT FOUND: {e}")

try:
    import bitsandbytes
    print(f"bitsandbytes={bitsandbytes.__version__}")
except ImportError as e:
    print(f"bitsandbytes NOT FOUND: {e}")
