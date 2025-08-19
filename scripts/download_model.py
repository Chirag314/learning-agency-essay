# scripts/download_model.py
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--out", default="models/microsoft-deberta-v3-base")
    parser.add_argument(
        "--revision", default=None, help="Optional: pin to a specific commit/tag"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[hf] snapshotting {args.model} -> {out}")
    snapshot_download(
        repo_id=args.model,
        local_dir=str(out),
        local_dir_use_symlinks=False,  # write real files on Windows
        revision=args.revision,  # or leave None for latest
        # You can ignore anything you don’t need; leaving default downloads all files:
        # ignore_patterns=["*.msgpack", "*.onnx", "tf_model.h5"],
    )
    print("[hf] done")


if __name__ == "__main__":
    main()
