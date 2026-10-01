import subprocess
import sys

try:
    result = subprocess.run(
        [sys.executable, "-c", "from transformers import AutoTokenizer; print('transformers OK')"],
        capture_output=True, text=True, timeout=15
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)
except Exception as e:
    print("Error:", e)
