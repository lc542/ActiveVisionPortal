import argparse, os, sys, numpy as np
from models.CLIPFamily import models, util

# Supported Models: OpenCLIP / CLIP, SigLIP, SigLIP2

# Examples:

# Single image embedding:
# python CLIP_main.py --backend openclip --task embed-image --image demo.jpg --model-name ViT-B-16 --pretrained openai --out runs/img_openai_vitb16.npy
# python CLIP_main.py --backend siglip --task embed-image --image demo.jpg --hf-model google/siglip2-base-patch16-224 --out runs/img_siglip2.npy

# Batch images from a directory:
# python CLIP_main.py --backend openclip --task embed-image --dir dataset/images --recursive --model-name ViT-g-14 --pretrained laion2b_s34b_b88k --out-dir runs/openclip_g14_laion/
# python CLIP_main.py --backend siglip --task embed-image --pattern "dataset/**/*.jpg" --hf-model google/siglip2-large-patch16-384 --out runs/siglip2_large_imgs.npy

# Text embedding:
# python CLIP_main.py --backend openclip --task embed-text --text "red car" --model-name ViT-B-16 --pretrained openai --out runs/txt_redcar.npy
# python CLIP_main.py --backend siglip --task embed-text --text-file queries.txt --hf-model google/siglip2-base-patch16-224 --out runs/queries_siglip2.npy

# List supported model weights:
# python CLIP_main.py --list-openclip
# python CLIP_main.py --list-siglip

def main():
    ap = argparse.ArgumentParser("Embed images/text with OpenCLIP or SigLIP/SigLIP2 (single/multi).")
    ap.add_argument("--backend", choices=["openclip", "siglip"])
    ap.add_argument("--task", choices=["embed-image", "embed-text"])

    # image inputs
    ap.add_argument("--image", type=str, help="single image path")
    ap.add_argument("--dir", type=str, help="directory of images")
    ap.add_argument("--recursive", action="store_true", help="recurse into subdirs with --dir")
    ap.add_argument("--pattern", type=str, help="glob pattern like 'data/**/*.jpg'")
    ap.add_argument("--list", type=str, help="text file with one image path per line")
    ap.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp")

    # text inputs
    ap.add_argument("--text", type=str)
    ap.add_argument("--text-file", type=str, help="text file with one query per line")

    # outputs
    ap.add_argument("--out", type=str, help="(matrix) output .npy (plus .json with ids)")
    ap.add_argument("--out-dir", type=str, help="(per-image/per-text) write one .npy per item")

    # general opts
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])

    # OpenCLIP opts
    ap.add_argument("--model-name", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--list-openclip", action="store_true",
                    help="list available OpenCLIP (model_name, pretrained) and exit")

    # SigLIP opts
    ap.add_argument("--hf-model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--list-siglip", action="store_true", help="print common SigLIP/SigLIP2 ids and exit")
    args = ap.parse_args()

    if args.list_openclip:
        try:
            import open_clip
            print("# OpenCLIP available (model_name, pretrained):")
            for name, pt in open_clip.list_pretrained():
                print(name, pt)
        except Exception as e:
            print("Failed to list OpenCLIP pretrained:", e, file=sys.stderr)
        return

    if args.list_siglip:
        common = [
            # SigLIP
            "google/siglip-base-patch16-224",
            "google/siglip-large-patch16-384",
            "google/siglip-so400m-patch14-384",
            # SigLIP2
            "google/siglip2-base-patch16-224",
            "google/siglip2-base-patch16-256",
            "google/siglip2-large-patch16-384",
            "google/siglip2-so400m-patch14-384",
            # NaFlex variants
            "google/siglip2-base-patch16-256-nap",
            "google/siglip2-large-patch16-384-nap",
        ]
        print("# Common SigLIP/SigLIP2 checkpoints:")
        for mid in common: print(mid)
        return

    if args.backend == "openclip":
        be = models.OpenCLIPBackend(args.model_name, args.pretrained, args.device, args.precision)
    else:
        be = models.HFSiglipBackend(args.hf_model, args.device, args.precision, args.load_in_4bit)

    if args.task == "embed-image":
        paths = util.iter_image_paths(args)
        imgs = util.load_images(paths)
        embs = be.encode_images(imgs, batch_size=args.batch_size)  # [N, D]
        if args.out_dir:
            util.save_per_image(args.out_dir, embs, paths)
            print(f"[OK] saved {len(paths)} image embeddings to {args.out_dir}")
        elif args.out:
            util.save_matrix(args.out, embs, paths)
            print(f"[OK] saved matrix to {args.out} (+ .json ids)")
        else:
            raise SystemExit("Please specify --out or --out-dir")

    else:  # embed-text
        texts = util.iter_texts(args)
        embs = be.encode_texts(texts, batch_size=max(8, args.batch_size * 4))  # texts are cheaper
        if args.out_dir:
            util.ensure_dir(args.out_dir)
            arr = embs.detach().cpu().numpy().astype(np.float32)
            for i, t in enumerate(texts):
                stem = f"{i:06d}"
                np.save(os.path.join(args.out_dir, f"{stem}.npy"), arr[i])
            with open(os.path.join(args.out_dir, "texts.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(texts))
            print(f"[OK] saved {len(texts)} text embeddings to {args.out_dir}")
        elif args.out:
            util.save_matrix(args.out, embs, list(texts))
            print(f"[OK] saved matrix to {args.out} (+ .json ids)")
        else:
            raise SystemExit("Please specify --out or --out-dir")


if __name__ == "__main__":
    main()
