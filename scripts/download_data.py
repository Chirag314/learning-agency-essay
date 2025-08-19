import argparse, os, zipfile, io, sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--competition", default="learning-agency-lab-automated-essay-scoring-2"
    )
    parser.add_argument("--out", default="data")
    parser.add_argument(
        "--only-metadata",
        action="store_true",
        help="Do not extract files; just test API auth.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("Install kaggle: pip install kaggle", file=sys.stderr)
        raise

    api = KaggleApi()
    api.authenticate()  # Uses env vars or ~/.kaggle/kaggle.json

    print(f"[kaggle] downloading files for competition: {args.competition}")
    buf = api.competition_download_files(args.competition, path=args.out, quiet=False)
    if args.only_metadata:
        print("[kaggle] metadata-only smoke done")
        return

    zip_path = Path(args.out) / f"{args.competition}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(args.out)
        zip_path.unlink(missing_ok=True)
    else:
        if isinstance(buf, bytes):
            with zipfile.ZipFile(io.BytesIO(buf), "r") as zf:
                zf.extractall(args.out)

    print(f"[kaggle] files extracted under: {args.out}")


if __name__ == "__main__":
    main()
