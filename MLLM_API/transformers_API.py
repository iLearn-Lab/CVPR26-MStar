from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, Qwen2_5_VLPreTrainedModel, Qwen2_5_VLModel, Qwen3VLForConditionalGeneration
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from transformers import StoppingCriteria, StoppingCriteriaList
import json
from fvcore.nn import FlopCountAnalysis, flop_count_table

from transformers import AutoProcessor
from PIL import Image


import time


class ResizingProcessor:
    def __init__(self, base_processor, longest_length=448):
        self.base_processor = base_processor
        self.longest_length = longest_length

    def __getattr__(self, name):
        return getattr(self.base_processor, name)

    def _resize_if_needed(self, image):
        width, height = image.size
        max_dim = max(width, height)

        if max_dim <= self.longest_length:
            return image

        scale = self.longest_length / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

    def __call__(self, *args, **kwargs):
        if 'images' in kwargs:
            imgs = kwargs['images']
            if isinstance(imgs, Image.Image):
                kwargs['images'] = self._resize_if_needed(imgs)
            elif isinstance(imgs, list):
                kwargs['images'] = [self._resize_if_needed(img) for img in imgs]

        return self.base_processor(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.base_processor.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.base_processor.batch_decode(*args, **kwargs)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_tokens, tokenizer):
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer

        self.encoded_stop_tokens = [tokenizer.encode(stop_token, add_special_tokens=False) for stop_token in stop_tokens]

    def __call__(self, input_ids, scores, **kwargs):
        for stop_token in self.encoded_stop_tokens:
            if len(input_ids[0]) >= len(stop_token):
                if all(input_ids[0][-len(stop_token)+i] == stop_token[i] for i in range(len(stop_token))):
                    return True

        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(text.endswith(stop_token) for stop_token in self.stop_tokens)

def load_transformers_model(model_ckpt, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="cuda", accelerator=None):
    processor = AutoProcessor.from_pretrained(model_ckpt, use_fast=True)

    if "Qwen2.5-VL" in model_ckpt:
        print("Loading Qwen2.5-VL model...")

        mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_ckpt,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map={
                "": accelerator.process_index

            },

        )
    elif "Qwen3-VL" in model_ckpt or "Qwen3VL" in model_ckpt:
        print("Loading Qwen3-VL model...")
        mllm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_ckpt,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map={
                "": accelerator.process_index

            },

        )
    else:
        mllm = AutoModel.from_pretrained(
            model_ckpt,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,

            )

    return processor, mllm

def generate_with_transformers_model(
    model,
    processor,
    conversation,
    temperature=0.6,
    max_new_tokens=1024,
    num_return_sequences=2,
    do_sample=True,
    repetition_penalty=1.0,
    top_p=0.7,
    top_k=20,
    stop_words=None,
    num_beams=1,
    ):
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",

    ).to(model.device)

    stopping_criteria = None
    if stop_words:
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_words, processor.tokenizer)])

    if hasattr(model, "module"):
        model = model.module

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        stopping_criteria=stopping_criteria,
    )

    assert len(inputs.input_ids) == 1, "Multiple different questions cannot be passed during the creation of a single node"

    generated_ids = [output[len(inputs.input_ids[0]):] for output in output_ids]
    output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return output_texts

def resize_image(input_path, longest_length = 800):
    with Image.open(input_path) as img:
        width, height = img.size

        if width <= longest_length and height <= longest_length:
            return img.copy()

        if width > height:
            new_width = longest_length
            new_height = int(height * (longest_length / width))
        else:
            new_height = longest_length
            new_width = int(width * (longest_length / height))

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img
