import torch
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
import open_clip


def device_of(name):
    return torch.device(name if name else ("cuda" if torch.cuda.is_available() else "cpu"))


def l2norm(x): return torch.nn.functional.normalize(x, dim=-1)


class OpenCLIPBackend:
    def __init__(self, model_name="ViT-B-16", pretrained="openai", device=None, precision="bf16"):
        self.device = device_of(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        if precision.lower() in ("fp16", "float16", "half"):
            self.model.half()
        elif precision.lower() in ("bf16", "bfloat16"):
            self.model.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image], batch_size=32) -> torch.Tensor:
        embs = []
        model_dtype = next(self.model.parameters()).dtype
        for i in range(0, len(pil_images), batch_size):
            batch = [self.preprocess(im) for im in pil_images[i:i + batch_size]]
            x = torch.stack(batch, dim=0).to(self.device, dtype=model_dtype)
            if next(self.model.parameters()).dtype == torch.float16: x = x.half()
            feats = self.model.encode_image(x)
            embs.append(l2norm(feats))
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size=128) -> torch.Tensor:
        embs = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            feats = self.model.encode_text(toks)
            embs.append(l2norm(feats))
        return torch.cat(embs, dim=0)


class HFSiglipBackend:

    def __init__(self, model_id, device=None, precision="bf16", load_in_4bit=False):
        self.device = device_of(device)
        torch_dtype = {"fp16": torch.float16, "float16": torch.float16,
                       "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(precision.lower(), torch.bfloat16)
        quant = {}
        if load_in_4bit:
            quant = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch_dtype, attn_implementation="sdpa", **quant
        ).eval()
        self.is_siglip2 = "siglip2" in model_id.lower()

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image], batch_size=32) -> torch.Tensor:
        embs = []
        model_dtype = next(self.model.parameters()).dtype
        for i in range(0, len(pil_images), batch_size):
            inputs = self.proc(images=pil_images[i:i + batch_size], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)
                feats = out.image_embeds
            embs.append(l2norm(feats))
        return torch.cat(embs, dim=0).to("cpu")

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size=128) -> torch.Tensor:
        embs = []
        for i in range(0, len(texts), batch_size):
            inputs = self.proc(text=texts[i:i + batch_size], return_tensors="pt", padding=True,
                               max_length=64 if self.is_siglip2 else None).to(self.model.device)
            if hasattr(self.model, "get_text_features"):
                feats = self.model.get_text_features(**inputs)
            else:
                out = self.model(**inputs, images=Image.new("RGB", (224, 224)))
                feats = out.text_embeds
            embs.append(l2norm(feats))
        return torch.cat(embs, dim=0).to("cpu")
