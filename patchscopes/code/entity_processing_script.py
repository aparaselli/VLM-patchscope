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

  ModelAndTokenizer,
  make_inputs,
  decode_tokens,
  find_token_range,
  predict_from_input,
)

from patchscopes_utils import *

from tqdm import tqdm
tqdm.pandas()



model_to_hook = {
    "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
    "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj
}


# Load model

model_name = "lmsys/vicuna-7b-v1.5"
sos_tok = False

if "13b" in model_name or "12b" in model_name:
    torch_dtype = torch.float16
else:
    torch_dtype = None

print("Loading model ")
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True, #while testing 
    torch_dtype=torch_dtype,
)
mt.set_hs_patch_hooks = model_to_hook[model_name]
mt.model.eval()
print("Model loaded")

df = pd.read_csv("/users/aparasel/scratch/temp_data/popQA.tsv", sep="\t")
print(f"df.columns is {df.columns}")

tmp = pd.concat([
    df[["subj", "s_wiki_title", "s_pop"]].drop_duplicates().sort_values(by="s_pop").head(200),
    df[["subj", "s_wiki_title", "s_pop"]].drop_duplicates().sort_values(by="s_pop").tail(200)
])
tmp.index = np.arange(tmp.shape[0])
tmp = tmp.iloc[371:373]#FOR TESTING
print(f"tmp.shape is {tmp.shape}")
print(tmp.head(2))




generation_mode = True
max_gen_len = 50

# prompt_source = "Devalan" # "Jurassic Park" # "Back to the Future" # "New York City" #"Muhammad Ali" # "Alexander the Great" #"Diana, Princess of Wales"
prompt_target = f"Syria: Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South Korean multinational major appliance and consumer electronics corporation, x"
position_source = -1
position_target = -1


for layer_source in range(10): #only trying one for now
    layer_target = layer_source

    tmp[f"inspect_layer{layer_source}"] = tmp.subj.progress_apply(
        lambda prompt_source: inspect(mt, prompt_source, prompt_target, layer_source, layer_target,
                                      position_source, position_target,
                                      generation_mode=generation_mode, max_gen_len=max_gen_len, verbose=False))

tmp.to_csv("popqa_200-200_input_processing_vicuna-7b-v1.1.csv", index=False)