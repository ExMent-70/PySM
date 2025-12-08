import sys
import os
try: import _core
except ImportError as e:
    print(f"CRITICAL ERROR: Security module '_core' missing."); sys.exit(1)
if __name__ == "__main__": _core.main()
