import os #SPECIFIC TO ATHU OSCAR
os.environ["HF_HOME"] = "/oscar/scratch/aparasel/hf_cache"
#os.environ["TRANSFORMERS_CACHE"] = "/oscar/scratch/aparasel/hf_cache/transformers"
#os.environ["DATASETS_CACHE"] = "/oscar/scratch/aparasel/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/oscar/scratch/aparasel/hf_cache/hub"


from ast import literal_eval
import functools
import json
import os
import random
import shutil
from PIL import Image

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
import datasets
torch.set_grad_enabled(False)

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook",
        rc={"font.size":16,
            "axes.titlesize":16,
            "axes.labelsize":16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style='whitegrid')

# Utilities

from general_utils import (
  LLaVAModelAndProcessor,
  ModelAndTokenizer,
  make_inputs,
  decode_tokens,
  find_token_range,
  predict_from_input,
)

from transformers import (
    AutoConfig,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,  # v1.6 path
)

from patchscopes_utils import *

from tqdm import tqdm
tqdm.pandas()


def load_celeba_df(images_dir, attr_path, as_bool=True, verify_files=True):
    """
    images_dir: directory containing the aligned images (e.g. .../img_align_celeba)
    attr_path:  path to list_attr_celeba.txt
    as_bool:    convert attributes from {-1,1} to {False, True}
    verify_files: drop rows whose image file is missing on disk
    """
    with open(attr_path, "r") as f:
        #_num_images = f.readline().strip()         # e.g., "202599" (not used)
        header_line = f.readline().strip()         # 40 attribute names (space-separated)

    attr_names = header_line.split()
    cols = ["image_id"] + attr_names

    df = pd.read_csv(
        attr_path,
        delim_whitespace=True,     # handles variable spacing
        skiprows=1,                # skip num-images and header line
        names=cols,
        engine="python",
    )

    if as_bool:
        df[attr_names] = (df[attr_names] == 1)     # {-1,1} -> {False,True}
    else:
        df[attr_names] = df[attr_names].astype("int8")

    df["image_path"] = df["image_id"].apply(lambda x: os.path.join(images_dir, x))

    return df

images_dir = "/users/aparasel/scratch/temp_data/img_align_celeba"
attr_path  = "/users/aparasel/scratch/temp_data/list_attr_celeba.txt"

celeba_df = load_celeba_df(images_dir, attr_path, as_bool=True, verify_files=True)
celeba_df = celeba_df.iloc[14:28] #FOR TESTING [:1000]
print(celeba_df.shape)
print(celeba_df.head())



model_to_hook = {
    "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
    "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj,
    "llava-hf/llava-v1.6-vicuna-7b-hf": set_hs_patch_hooks_llama
}


# Load model

model_name = "llava-hf/llava-v1.6-vicuna-7b-hf" #base is lmsys/vicuna-7b-v1.5
sos_tok = False

torch_dtype = torch.float16

print("Loading model ")
mt = LLaVAModelAndProcessor(
    model_name,
    low_cpu_mem_usage=True,  # while testing
    torch_dtype=torch_dtype,
)
mt.set_hs_patch_hooks = set_hs_patch_hooks_llama
mt.model.eval()
print("LLAVA Model loaded :)")






generation_mode = True
max_gen_len = 50

#load processor 
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

chat = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "X X X X X"},
        ],
    },
    {
      "role": "Assistant",
      "content": [
          {"type": "text", "text": "Sure, I’ll summarize your message:"},
        ],
    },
]

chat_source = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "{Black, Blond, Brown, Gray, Red, NONE} Out of the listed hair colors which one best describes the person in the image?"},
          {"type": "image"},
        ],
    },
]



# prompt_source = "Devalan" # "Jurassic Park" # "Back to the Future" # "New York City" #"Muhammad Ali" # "Alexander the Great" #"Diana, Princess of Wales"
prompt_target = processor.apply_chat_template(
    chat,
    tokenize=False,
    add_generation_prompt=False
)

print("PROMPT TARGET:")
print(prompt_target)

position_target = 4 #replace the Xs

prompt_source = processor.apply_chat_template(
    chat_source,
    tokenize=False,
    add_generation_prompt=True
)#"<image>\n{Black, Blond, Brown, Gray, Red, NONE} Out of the listed hair colors the one that best describes this person is"

print("PROMPT SOURCE:")
print(prompt_source)

inp_source = make_inputs(mt.tokenizer, [prompt_source], device=mt.device, model=mt, images=Image.open(celeba_df['image_path'].iloc[0]).convert("RGB"))
source_tokens = [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]]
print("TOKENS")
print(source_tokens)
img_loc = 5
num_tokens = 5
for i in range(1,num_tokens+1):
    source_tokens.insert(5,f"image_token-{i}")

celeba_df['prompt_source'] = [prompt_source for k in range(celeba_df.shape[0])]
celeba_df['prompt_target'] = [prompt_target for k in range(celeba_df.shape[0])]

layer_target = 2
parts = []
for i in range(-6,(-1*len(source_tokens))+4,-1):#range(-1,(-1*len(source_tokens))-1,-1)
    position_source = i
        
    temp_celeba_df = celeba_df[['image_id','image_path','prompt_source','prompt_target']].copy()
    temp_celeba_df['position_source'] = [position_source for k in range(celeba_df.shape[0])]
    temp_celeba_df['token_from_source'] = [source_tokens[position_source] for k in range(celeba_df.shape[0])]
    print(f"Starting position {i}")


    for layer_source in range(32): #32
        

        temp_celeba_df[f"inspect_layer{layer_source}"] = temp_celeba_df.image_path.progress_apply(
            lambda img_path: inspect(mt, prompt_source, prompt_target, layer_source, layer_target,
                                        position_source, position_target,
                                        generation_mode=generation_mode, max_gen_len=max_gen_len, verbose=False,image_source=Image.open(img_path).convert("RGB"),selfIE=5))

    #TODO: ADD temp_celeba_df to running dataframe
    parts.append(temp_celeba_df)
    
final_df = pd.concat(parts, ignore_index=True)     
final_df.to_csv("hair-color-exploration-2-CELEB-A-selfie-interp_llava-7b.csv", index=False) #TODO: Save running dataframe instead of celeba_df
