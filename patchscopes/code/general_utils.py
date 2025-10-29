# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re

import torch
import transformers
from transformers import AutoProcessor
from transformers import (
    AutoConfig,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,  # v1.6 path
)

class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      use_fast=True,
      device="cuda",
      ):
    if tokenizer is None:
      assert model_name is not None
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    if model is None:
      assert model_name is not None
      model = transformers.AutoModelForCausalLM.from_pretrained(
          model_name, low_cpu_mem_usage=low_cpu_mem_usage,
          torch_dtype=torch_dtype, device_map="auto" #automatically put it on device
          )
      #if device is not None: #remove bc i dont have enough space
      #  model.to(device)
      set_requires_grad(False, model)
      model.eval()
    self.tokenizer = tokenizer
    self.model = model
    self.device = device
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, model=None, images=None, device="cuda"):
  """Prepare inputs to the model."""
  if images is not None:
    assert model is not None, "Must pass in model to make_inputs for multimodal inputs"
    return model.make_inputs(prompts, images=images)
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)


class LLaVAModelAndProcessor(ModelAndTokenizer): #LLAVA subclass - Athu 
    """
    LLaVA-specific subclass:
      - loads LlavaForConditionalGeneration + AutoProcessor
      - exposes tokenizer via processor.tokenizer (so your code keeps working)
      - publishes layer_names for Vicuna decoder inside LLaVA
    """
    def __init__(self, model_name, low_cpu_mem_usage=False, torch_dtype=torch.float16,
                 use_fast=True, device="cuda"):
        assert LlavaNextForConditionalGeneration is not None, "Install transformers w/ LLaVA support."

        # Load image processor
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        tokenizer = self.processor.tokenizer

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        set_requires_grad(False, model)
        model.eval()

        # Initialize base class with our concrete model/tokenizer
        super().__init__(model_name=model_name, model=model, tokenizer=tokenizer,
                         low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype,
                         use_fast=use_fast, device=device)

        # Override layer discovery to point at the Vicuna decoder inside LLaVA
        # (language_model -> model -> layers)
        layers = self.model.language_model.model.layers
        self.layer_names = [f"language_model.model.layers.{i}" for i in range(len(layers))]
        self.num_layers = len(self.layer_names)

    # Convenience: unified accessors for hook code
    @property
    def decoder_layers(self):
        return self.model.language_model.model.layers

    @property
    def decoder_norm(self):
        return self.model.language_model.model.norm

    def make_inputs(self, prompts, images=None):
        """
        Multimodal-safe inputs:
          - text-only (images=None): returns {input_ids, attention_mask}
          - image+text: returns {input_ids, attention_mask, pixel_values}
        """
        if images is None:
            batch = self.processor(text=prompts, return_tensors="pt", padding=False)
        else:
            batch = self.processor(text=prompts, images=images, return_tensors="pt", padding=False)
        return {k: v.to(self.device) for k, v in batch.items()}