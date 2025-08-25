import os
import re
import json
import argparse
import traceback
from glob import glob
from typing import Dict, List, Iterable
from PIL import Image
from tqdm import tqdm
import torch
import platform
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForImageTextToText
)
from pprint import pprint


def list_models_str(models_arg: str) -> List[str]:
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def iter_images(img_dir: str) -> Iterable[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for p in sorted(glob(os.path.join(img_dir, "*"))):
        if p.lower().endswith(exts):
            yield p


def chunked(lst: List[str], bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i: i + bs]


def clean_assistant_text(decoded: str) -> str:
    s = decoded.replace("\r\n", "\n").replace("\r", "\n").strip()

    m = re.search(r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)", s, flags=re.S | re.I)
    if m:
        out = m.group(1)
        out = re.sub(r"<\|.*?\|>", "", out)
        return out.strip()

    m = re.search(r"(?:^|\n)assistant\s*[:\n]\s*(.*)", s, flags=re.S | re.I)
    if m:
        seg = m.group(1)
        seg = re.split(r"(?:^|\n)(?:system|user|assistant)\s*[:\n]", seg, maxsplit=1, flags=re.I)[0]
        return seg.strip()

    s2 = re.sub(r"^(?:system|user)\s*[:\n].*?(?=(?:^|\n)assistant\s*[:\n])", "", s, flags=re.S | re.I)
    if s2 != s:
        m = re.search(r"(?:^|\n)assistant\s*[:\n]\s*(.*)", s2, flags=re.S | re.I)
        if m:
            seg = re.split(r"(?:^|\n)(?:system|user|assistant)\s*[:\n]", m.group(1), maxsplit=1, flags=re.I)[0]
            return seg.strip()

    s = re.sub(r"<\|.*?\|>", "", s)
    return s.strip()


def probe_model(model_id: str) -> Dict:
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        arch = getattr(cfg, "architectures", None)
        try:
            _ = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            proc_ok = True
        except Exception:
            proc_ok = False
        return {"ok": True, "arch": arch, "processor_ok": proc_ok, "reason": ""}
    except Exception as e:
        return {"ok": False, "arch": None, "processor_ok": False, "reason": str(e)}


def _auto_max_memory(reserve_gib=2):
    mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GiB
            cap = max(1, int(total) - reserve_gib)  # 预留 reserve_gib GiB
            mem[i] = f"{cap}GiB"
    return mem or None


def build_common_kwargs(dtype: torch.dtype, quant: str):
    common = dict(
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=_auto_max_memory(reserve_gib=2),
    )

    use_flash = False
    if platform.system() == "Linux" and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            use_flash = True
        except Exception:
            use_flash = False

    if use_flash:
        common["attn_implementation"] = "flash_attention_2"
    else:
        common["attn_implementation"] = "sdpa"

    if quant == "8bit":
        common.update(dict(load_in_8bit=True))
    elif quant == "4bit":
        common.update(
            dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        )
    return common


def load_model_and_processor(name: str, dtype: torch.dtype, quant: str = "none"):
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)

    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    is_vl = ("vl" in model_type) or ("vision" in model_type) or ("multimodal" in model_type)

    common = build_common_kwargs(dtype, quant)
    try:
        m = AutoModelForVision2Seq.from_pretrained(name, **common).eval()
        return m, proc, "v2s"
    except ImportError as e:
        msg = str(e)
        if "flash_attn" in msg or "FlashAttention2" in msg:
            common_sdpa = build_common_kwargs(dtype, quant)
            common_sdpa["attn_implementation"] = "sdpa"
            m = AutoModelForVision2Seq.from_pretrained(name, **common_sdpa).eval()
            return m, proc, "v2s"
        v2s_err = e
    except Exception as e:
        v2s_err = e

    try:
        m = AutoModelForImageTextToText.from_pretrained(name, **common).eval()
        return m, proc, "itt"
    except Exception as e_itt:
        itt_err = e_itt

    if not is_vl:
        try:
            m = AutoModelForCausalLM.from_pretrained(name, **common).eval()
            return m, proc, "causal"
        except Exception as e_causal:
            causal_err = e_causal
            raise RuntimeError(
                f"load failed for {name} -> Vision2Seq({v2s_err}) | ImageTextToText({itt_err}) | CausalLM({causal_err})"
            )

    raise RuntimeError(
        f"load failed for {name} -> Vision2Seq({v2s_err}) | ImageTextToText({itt_err})"
    )


def _to_dev_like_model(model, inputs: Dict):
    dev = next(model.parameters()).device
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _build_inputs(model, proc, imgs: List[Image.Image], prompt: str):
    name = (getattr(getattr(model, "config", None), "name_or_path", "") or "").lower()
    if "qwen2.5-vl" in name or ("qwen" in name and "vl" in name):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True)
    else:
        inputs = proc(text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True)
    return inputs


def _gen_kwargs(proc, max_new: int):
    tok = getattr(proc, "tokenizer", None)
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    return dict(
        max_new_tokens=max(128, max_new),
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )


def generate_batch(model, proc, imgs: List[Image.Image], prompt: str, max_new: int) -> List[str]:
    inputs = _build_inputs(model, proc, imgs, prompt)
    inputs = _to_dev_like_model(model, inputs)
    gen_kwargs = _gen_kwargs(proc, max_new)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = proc.batch_decode(out, skip_special_tokens=False)
    return [clean_assistant_text(t) for t in decoded]


def generate_any(model, proc, img: Image.Image, prompt: str, max_new: int) -> str:
    inputs = _build_inputs(model, proc, [img], prompt)
    inputs = _to_dev_like_model(model, inputs)
    gen_kwargs = _gen_kwargs(proc, max_new)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = proc.batch_decode(out, skip_special_tokens=False)[0]
    return clean_assistant_text(decoded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="Comma-separated HF model ids")
    ap.add_argument("--img_dir", default="images")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16"])
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--prompt", default="Describe the image briefly.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--probe-only", action="store_true", help="Detect model availability.")
    args = ap.parse_args()

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    models = list_models_str(args.models)

    if args.probe_only:
        report = {m: probe_model(m) for m in models}
        print(json.dumps({"probe": report}, ensure_ascii=False, indent=2))
        return

    dtype = torch.bfloat16 if args.dtype.lower() in ["bf16", "bfloat16"] else torch.float16

    paths = list(iter_images(args.img_dir))
    if not paths:
        raise SystemExit(f"No images found in {args.img_dir}")

    for name in models:
        print(f"\n=== [{name}] ===", flush=True)
        out_path = f"captions_{name.split('/')[-1]}.jsonl"
        try:
            model, proc, typ = load_model_and_processor(name, dtype, quant=args.quant)

            dm = getattr(model, "hf_device_map", None)
            print("\n[hf_device_map] ↓")
            pprint(dm, width=120)
            if dm:
                used = sorted({str(v) for v in dm.values()})
                print("[hf_device_map] devices used:", used)
                try:
                    if torch.cuda.device_count() >= 4:
                        assert len(
                            used) >= 4, "Model was not sharded across 4 GPUs. Consider lowering max_memory per GPU."
                except Exception as _e:
                    print("[warn] device assertion skipped:", _e)
        except Exception as e:
            print(f"[ERROR] load {name} failed: {e}")
            traceback.print_exc()
            continue

        with open(out_path, "w", encoding="utf-8") as fout:
            for paths_chunk in tqdm(list(chunked(paths, args.batch)), desc=f"{typ}-caption"):
                imgs = [Image.open(p).convert("RGB") for p in paths_chunk]
                try:
                    texts = generate_batch(model, proc, imgs, args.prompt, args.max_new)
                except Exception as e:
                    texts = []
                    for img in imgs:
                        try:
                            texts.append(generate_any(model, proc, img, args.prompt, args.max_new))
                        except Exception as e2:
                            texts.append(f"[ERROR] {e2}")

                for p, txt in zip(paths_chunk, texts):
                    fout.write(json.dumps({"model": name, "path": p, "text": txt}, ensure_ascii=False) + "\n")

        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
