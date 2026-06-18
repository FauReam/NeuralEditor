"""Download torch CUDA wheel with retry."""
import os
import time
import urllib.request

URL = (
    "https://download.pytorch.org/whl/cu121/"
    "torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
)
PATH = r"C:\Users\46326\AppData\Local\Temp\torch_cu121.whl"

print(f"Downloading: {URL}")
print(f"Target: {PATH}")

for i in range(5):
    try:
        start = time.time()
        urllib.request.urlretrieve(URL, PATH)
        elapsed = time.time() - start
        size = os.path.getsize(PATH) / 1e9  # noqa: F821
        speed = size * 1024 / elapsed
        print(f"DONE: {size:.1f} GB in {elapsed:.0f}s ({speed:.1f} MB/s)")
        break
    except Exception as e:
        print(f"Retry {i+1}/5: {e}")
        time.sleep(2)
