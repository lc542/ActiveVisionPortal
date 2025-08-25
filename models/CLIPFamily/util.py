import os, sys, glob, json, numpy as np, torch
from PIL import Image
from typing import List


def ensure_dir(p):
    if p: os.makedirs(p, exist_ok=True)


def iter_image_paths(args) -> List[str]:
    paths = []
    exts = tuple(args.exts.split(",")) if args.exts else (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if args.image:
        paths = [args.image]
    elif args.pattern:
        paths = sorted(glob.glob(args.pattern, recursive=True))
    elif args.dir:
        for root, _, files in os.walk(args.dir):
            for f in files:
                if f.lower().endswith(exts):
                    paths.append(os.path.join(root, f))
            if not args.recursive: break
        paths = sorted(paths)
    elif args.list:
        with open(args.list, "r", encoding="utf-8") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
    else:
        raise SystemExit("No image input. Use --image / --dir / --pattern / --list")
    if len(paths) == 0:
        raise SystemExit("No images found.")
    return paths


def iter_texts(args) -> List[str]:
    if args.text is not None:
        return [args.text]
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    else:
        raise SystemExit("No text provided. Use --text or --text-file.")


def load_images(paths: List[str]) -> List[Image.Image]:
    ims = []
    for p in paths:
        try:
            ims.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[WARN] fail to open {p}: {e}", file=sys.stderr)
    if len(ims) == 0: raise SystemExit("No valid images after loading.")
    return ims


def save_matrix(out_path: str, mat: torch.Tensor, ids: List[str]):
    ensure_dir(os.path.dirname(out_path))
    arr = mat.detach().to(dtype=torch.float32, device="cpu").numpy()
    np.save(out_path, arr)
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"ids": ids}, f, ensure_ascii=False, indent=2)


def save_per_image(out_dir: str, mat: torch.Tensor, paths: List[str]):
    ensure_dir(out_dir)
    arr = mat.detach().to(dtype=torch.float32, device="cpu").numpy()
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), arr[i])
