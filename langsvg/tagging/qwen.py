"""Batched JSON generation with Qwen3.5 vision-language models (Hugging Face transformers)."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen3.5-27B"

# One request = {"system_prompt": str, "user_payload": dict, "full_image": Image | None,
#                "crop_image": Image | None, "extra_images": [{"image": Image, "description": str}]}
Request = Dict[str, Any]
Output = Tuple[Dict[str, Any], str]  # (parsed JSON object, raw decoded text)

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    return re.sub(r"(?is)<think>.*?</think>\s*", "", text or "").strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model text, tolerating code fences and surrounding prose."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_messages(request: Request) -> List[Dict[str, Any]]:
    """Chat messages for one request: images first (with numbered notes), then the JSON payload."""
    user_payload = request["user_payload"]
    content: List[Dict[str, Any]] = []
    notes: List[str] = []
    index = 1
    if request.get("full_image") is not None:
        content.append({"type": "image", "image": request["full_image"]})
        notes.append(f"Image {index} is the original full scene.")
        index += 1
    if request.get("crop_image") is not None:
        content.append({"type": "image", "image": request["crop_image"]})
        description = str(
            user_payload.get("image_2_description") or "This image is an additional crop around the target bbox."
        )
        notes.append(f"Image {index}: {description}")
        index += 1
    for item in request.get("extra_images") or []:
        image = item.get("image") if isinstance(item, dict) else None
        if image is None:
            continue
        content.append({"type": "image", "image": image})
        notes.append(f"Image {index}: " + str(item.get("description") or "This image is an additional reference image."))
        index += 1
    text = "\n".join(notes)
    if text:
        text += "\n\n"
    text += "Use this target-node metadata and return JSON only:\n" + json.dumps(user_payload, ensure_ascii=False, indent=2)
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": str(request["system_prompt"])},
        {"role": "user", "content": content},
    ]


class QwenVLGenerator:
    """Greedy (or sampled) batched JSON generation with a Qwen3.5 VLM.

    Args:
        model: Hugging Face model id or local snapshot directory.
        dtype: ``auto``, ``bfloat16``, ``float16`` or ``float32``.
        device_map: passed to ``from_pretrained`` (``auto``, ``cuda:0``, ...).
        max_pixels / min_pixels: image processor resolution bounds.
        attn_implementation: optional attention backend (e.g. ``sdpa``).
        enable_thinking: forwarded to the chat template.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_pixels: Optional[int] = 401408,
        min_pixels: Optional[int] = None,
        attn_implementation: str = "",
        enable_thinking: bool = False,
    ) -> None:
        import torch
        import transformers

        self.torch = torch
        self.model_name = model
        self.enable_thinking = enable_thinking
        self._template_accepts_thinking = True

        processor_kwargs: Dict[str, Any] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        self.processor = transformers.AutoProcessor.from_pretrained(model, **processor_kwargs)
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
        model_kwargs: Dict[str, Any] = {
            "device_map": self._parse_device_map(device_map),
            "low_cpu_mem_usage": True,
            "dtype": self._parse_dtype(dtype),
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = model_cls.from_pretrained(model, **model_kwargs)
        self.model.eval()
        self.input_device = self._first_device()

    def _parse_dtype(self, name: str):
        name = str(name or "auto").strip().lower()
        if name == "auto":
            return "auto"
        return getattr(self.torch, _DTYPE_ALIASES.get(name, name))

    def _parse_device_map(self, device_map: str):
        text = str(device_map or "auto").strip()
        lowered = text.lower()
        if lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return lowered
        if lowered.isdigit():
            return int(lowered)
        if lowered.startswith("cuda") or lowered in {"cpu", "mps"}:
            return self.torch.device(text)
        return device_map

    def _first_device(self):
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, int):
                    return self.torch.device(f"cuda:{value}")
                if isinstance(value, str) and value not in {"cpu", "disk", "meta"}:
                    return self.torch.device(value)
        return next(self.model.parameters()).device

    def _render_prompt(self, messages: List[Dict[str, Any]]) -> str:
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._template_accepts_thinking:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.processor.apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            kwargs.pop("enable_thinking")
            self._template_accepts_thinking = False
            return self.processor.apply_chat_template(messages, **kwargs)

    def _generation_kwargs(
        self, max_new_tokens: int, temperature: float, top_p: float, top_k: int, repetition_penalty: float
    ) -> Dict[str, Any]:
        do_sample = temperature > 0.0
        kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        generation_config = copy.deepcopy(self.model.generation_config)
        if not do_sample:
            # Neutralize the checkpoint's sampling defaults so greedy decoding does not warn.
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
        kwargs["generation_config"] = generation_config
        tokenizer = self.processor.tokenizer
        if tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.eos_token_id
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            if top_k > 0:
                kwargs["top_k"] = top_k
        if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6:
            kwargs["repetition_penalty"] = repetition_penalty
        return kwargs

    def generate_json_batch(
        self,
        requests: Sequence[Request],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
    ) -> List[Output]:
        """Run one padded batch; raises ``RuntimeError`` if any response is not a JSON object."""
        if not requests:
            return []
        prompts: List[str] = []
        images: List[Image.Image] = []
        for request in requests:
            messages = build_messages(request)
            prompts.append(self._render_prompt(messages))
            images.extend(item["image"] for item in messages[1]["content"] if item["type"] == "image")
        processor_kwargs: Dict[str, Any] = {"text": prompts, "return_tensors": "pt", "padding": True}
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        input_ids = inputs["input_ids"]
        inputs = inputs.to(self.input_device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, **self._generation_kwargs(max_new_tokens, temperature, top_p, top_k, repetition_penalty)
            )
        trimmed = [out[len(prompt) :].detach().cpu() for prompt, out in zip(input_ids, generated)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs: List[Output] = []
        for index, text in enumerate(decoded):
            result = extract_json_object(strip_thinking(str(text)))
            if result is None:
                raise RuntimeError(f"Transformers batch response {index} was not valid JSON: {text}")
            outputs.append((result, text))
        return outputs

    def clear_cache(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()


ItemResult = Union[Tuple[Any, int], BaseException]  # (value, effective batch size) or the final error


def run_with_batch_splitting(
    run: Callable[[List[Any]], List[Any]],
    items: List[Any],
    clear_cache: Callable[[], None] = lambda: None,
    single_item_retries: int = 0,
) -> List[ItemResult]:
    """Call ``run(items)``; on any failure (CUDA OOM, invalid JSON, ...) split the batch in half and recurse.

    A single item that still fails is retried ``single_item_retries`` more times before its
    exception is returned in its slot. Returns one entry per item, in order.
    """
    if not items:
        return []
    try:
        outputs = run(items)
        if len(outputs) != len(items):
            raise RuntimeError(f"Batch returned {len(outputs)} outputs for {len(items)} inputs")
        return [(output, len(items)) for output in outputs]
    except Exception as exc:  # noqa: BLE001 - any failure triggers the split/retry fallback
        clear_cache()
        if len(items) > 1:
            mid = len(items) // 2
            return run_with_batch_splitting(run, items[:mid], clear_cache, single_item_retries) + run_with_batch_splitting(
                run, items[mid:], clear_cache, single_item_retries
            )
        error: BaseException = exc
    for _ in range(single_item_retries):
        try:
            return [(run(items)[0], 1)]
        except Exception as exc:  # noqa: BLE001
            clear_cache()
            error = exc
    return [error]
