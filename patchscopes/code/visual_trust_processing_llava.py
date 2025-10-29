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
celeba_df = celeba_df.iloc[:1000] #FOR TESTING [:1000]
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

# prompt_source = "Devalan" # "Jurassic Park" # "Back to the Future" # "New York City" #"Muhammad Ali" # "Alexander the Great" #"Diana, Princess of Wales"
prompt_target = "On a scale of 1-10 the trustworthiness of this person is a"#f"On a scale of 1-10 the trustworthiness of this person is a x"
position_source = -1
position_target = -1
prompt_source = "<image>\nOn a scale of 1-10 the trustworthiness of this person is a"

for layer_source in range(32): #only trying one for now
    layer_target = layer_source

    celeba_df[f"inspect_layer{layer_source}"] = celeba_df.image_path.progress_apply(
        lambda img_path: inspect(mt, prompt_source, prompt_target, layer_source, layer_target,
                                      position_source, position_target,
                                      generation_mode=generation_mode, max_gen_len=max_gen_len, verbose=False,image_source=Image.open(img_path).convert("RGB")))

celeba_df.to_csv("large-ident-CELEB_A-trustworthy_llava-7b-v1.1.csv", index=False)