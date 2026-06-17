"""Download Qwen2.5-7B-Instruct GGUF from HuggingFace."""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-7b-instruct-q4_k_m.gguf"


def download(
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILE,
    local_dir: str = "models",
) -> Path:
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Saved to: {path}")
    return Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--dir", default="models")
    args = parser.parse_args()
    download(args.repo, args.file, args.dir)
