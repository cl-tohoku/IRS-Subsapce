import os
import random
import math
import sys
import json
import numpy as np
from functools import partial
from collections import defaultdict

import torch
from baukit import TraceDict, nethook
from einops import rearrange, einsum
from peft import PeftModel
from datasets import Dataset as Dataset_ds
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2Tokenizer

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import sklearn
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.utils import shuffle

from sklearn.decomposition import PCA
from sklearn import linear_model

from typing import Callable, Any, Optional

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(parent_dir)

def compute_prev_clue_pos(input_ids, clue_id, last=False):
    """
    Computes the position of the previous query clue label token.
    Args:
        input_ids: input ids of the example.
        clue_id: clue id of the example.
    """
    prev_clue_pos = (
        (input_ids[:] == clue_id).nonzero().squeeze()
    ).cpu()
    
    if last:
        if prev_clue_pos.shape == torch.Size([]):
            prev_clue_pos = prev_clue_pos.view(-1)
        else:
            prev_clue_pos = prev_clue_pos[-1:]
    else:
        if prev_clue_pos.shape == torch.Size([]):
            prev_clue_pos = prev_clue_pos.view(-1)
        prev_clue_pos = prev_clue_pos[:1]
    return prev_clue_pos


def get_model_and_tokenizer(model_name, device):
    """
    Loads the model and tokenizer.
    Args:
        model_name (str): Name of the model to load.
    """

    if model_name == "llama":
        #model_id = "meta-llama/Meta-Llama-3-8B"
        #model_id = "meta-llama/Llama-3.1-8B"
        #model_id = "meta-llama/Llama-3.1-8B-Instruct"
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_name == "qwen":
        model_id = "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_name == "mistral":
        #model_id = "mistralai/Mistral-7B-v0.1"
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,)
    return model, tokenizer


def combine_data(list_data_file, sfout):
    data = []
    for data_file in list_data_file:
        with open(data_file, encoding="utf-8") as f:
            data_ = [json.loads(line) for line in f]
            data.extend(data_)

    fout = open(sfout, 'w')
    for row in data:
        fout.write(json.dumps(row))
        fout.write("\n")
    fout.close()


def ablate_table(input_t, att):
    input_t = input_t.replace(att, " ")
    return input_t


def load_tb_data(tokenizer, num_samples, data_file, rand=True):
    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    if rand:
        random.shuffle(data)

    ents1 = []
    ents2 = []
    ents3 = []
    
    atts1_1 = []
    atts1_2 = []
    atts1_3 = []
    atts2_1 = []
    atts2_2 = []
    atts2_3 = []
    atts3_1 = []
    atts3_2 = []
    atts3_3 = []
    atts4_1 = []
    atts4_2 = []
    atts4_3 = []

    labels_x = []
    labels_y = []
    labels_z = []
    
    prompts_temp = []
    prompts_table = []
    prompts_story = []
    
    for i in range(num_samples):
        ent1 = data[i]["ents"][0]
        ent2 = data[i]["ents"][1]
        ent3 = data[i]["ents"][2]
        ents1.append(tokenizer.encode(" %s"%ent1)[-1])
        ents2.append(tokenizer.encode(" %s"%ent2)[-1])
        ents3.append(tokenizer.encode(" %s"%ent3)[-1])

        att1_1 = data[i]["atts1"][0]
        att1_2 = data[i]["atts1"][1]
        att1_3 = data[i]["atts1"][2]
        atts1_1.append(tokenizer.encode(" %s"%att1_1)[-1])
        atts1_2.append(tokenizer.encode(" %s"%att1_2)[-1])
        atts1_3.append(tokenizer.encode(" %s"%att1_3)[-1])

        att2_1 = data[i]["atts2"][0]
        att2_2 = data[i]["atts2"][1]
        att2_3 = data[i]["atts2"][2]
        atts2_1.append(tokenizer.encode(" %s"%att2_1)[-1])
        atts2_2.append(tokenizer.encode(" %s"%att2_2)[-1])
        atts2_3.append(tokenizer.encode(" %s"%att2_3)[-1])

        att3_1 = data[i]["atts3"][0]
        att3_2 = data[i]["atts3"][1]
        att3_3 = data[i]["atts3"][2]
        atts3_1.append(tokenizer.encode(" %s"%att3_1)[-1])
        atts3_2.append(tokenizer.encode(" %s"%att3_2)[-1])
        atts3_3.append(tokenizer.encode(" %s"%att3_3)[-1])

        att4_1 = data[i]["atts4"][0]
        att4_2 = data[i]["atts4"][1]
        att4_3 = data[i]["atts4"][2]
        atts4_1.append(tokenizer.encode(" %s"%att4_1)[-1])
        atts4_2.append(tokenizer.encode(" %s"%att4_2)[-1])
        atts4_3.append(tokenizer.encode(" %s"%att4_3)[-1])
        
        ctx = "Context: " + data[i]["input"].strip()
        ctx_t = "Context: " + data[i]["t_input"].strip()
        ctx_s = "Context: " + data[i]["s_input"].strip()
        
        prompts_table.append(ctx_t)
        prompts_temp.append(ctx)
        prompts_story.append(ctx_s)
        
        ctx = "Context: " + data[i][f"input_{vari}"].strip()
        ctx_t = "Context: " + data[i][f"t_input_{vari}"].strip()
        ctx_s = "Context: " + data[i][f"s_input_{vari}"].strip()

        label_y = [1, 2, 3,]
        label_x = [1, 2, 3, 4, 5]
        label_z = [1, 2, 3, 4, 5]
        
        labels_y.append(label_y)
        labels_x.append(label_x)
        labels_z.append(label_x)
        
        ##
        
    input_tokens_table = tokenizer(prompts_table, padding=True, return_tensors="pt")
    input_ids_table = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp, padding=True, return_tensors="pt")
    input_ids_temp = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story, padding=True, return_tensors="pt")
    input_ids_story = input_tokens_story["input_ids"]
    
    ents1 = torch.tensor(ents1)
    ents2 = torch.tensor(ents2)
    ents3 = torch.tensor(ents3)

    atts1_1 = torch.tensor(atts1_1)
    atts1_2 = torch.tensor(atts1_2)
    atts1_3 = torch.tensor(atts1_3)
    atts2_1 = torch.tensor(atts2_1)
    atts2_2 = torch.tensor(atts2_2)
    atts2_3 = torch.tensor(atts2_3)
    atts3_1 = torch.tensor(atts3_1)
    atts3_2 = torch.tensor(atts3_2)
    atts3_3 = torch.tensor(atts3_3)
    atts4_1 = torch.tensor(atts4_1)
    atts4_2 = torch.tensor(atts4_2)
    atts4_3 = torch.tensor(atts4_3)

    labels_x = torch.tensor(labels_x)
    labels_y = torch.tensor(labels_y)
    labels_z = torch.tensor(labels_z)
    
    return (input_ids_table, input_ids_temp, input_ids_story,
            ents1, ents2, ents3,
            atts1_1, atts1_2, atts1_3,
            atts2_1, atts2_2, atts2_3,
            atts3_1, atts3_2, atts3_3,
            atts4_1, atts4_2, atts4_3,
            labels_x, labels_y, labels_z,)


def load_dataloader(
        tokenizer: AutoTokenizer,
        data_file: str,
        num_samples: int,
        batch_size: int,
        ablate_t: bool=False,):
    
    raw_data = load_tb_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=data_file,
        ablate_t=ablate_t,
    )

    base_tokens_table = raw_data[0]
    base_tokens_temp = raw_data[1]
    base_tokens_story = raw_data[2]

    ents1 = raw_data[3]
    ents2 = raw_data[4]
    ents3 = raw_data[5]

    atts1_1 = raw_data[6]
    atts1_2 = raw_data[7]
    atts1_3 = raw_data[8]

    atts2_1 = raw_data[9]
    atts2_2 = raw_data[10]
    atts2_3 = raw_data[11]

    atts3_1 = raw_data[12]
    atts3_2 = raw_data[13]
    atts3_3 = raw_data[14]

    atts4_1 = raw_data[15]
    atts4_2 = raw_data[16]
    atts4_3 = raw_data[17]

    labels_x = raw_data[18]
    labels_y = raw_data[19]
    labels_z = raw_data[20]

    dataset = Dataset_ds.from_dict(
        {
            "base_tokens_table": base_tokens_table,
            "base_tokens_temp": base_tokens_temp,
            "base_tokens_story": base_tokens_story,
            "ents1": ents1,
            "ents2": ents2,
            "ents3": ents3,
            "atts1_1": atts1_1,
            "atts1_2": atts1_2,
            "atts1_3": atts1_3,
            "atts2_1": atts2_1,
            "atts2_2": atts2_2,
            "atts2_3": atts2_3,
            "atts3_1": atts3_1,
            "atts3_2": atts3_2,
            "atts3_3": atts3_3,
            "atts4_1": atts4_1,
            "atts4_2": atts4_2,
            "atts4_3": atts4_3,
            "labels_x": labels_x,
            "labels_y":	labels_y,
            "labels_z":	labels_z,
        }
    ).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def get_x_y(
        model: AutoModelForCausalLM,
        dataloader: torch.utils.data.DataLoader,
        layer: int=15,
        input_tp: str="table"):
    """
    input_tp: "table", "temp", "story"

    """
    hook_points = [
        f"model.layers.{layer}"
    ]
    apply_softmax = torch.nn.Softmax(dim=-1)
    xs_ents1 = []
    xs_ents2 = []
    xs_ents3 = []

    xs_atts1_1 = []
    xs_atts1_2 = []
    xs_atts1_3 = []
    
    xs_atts2_1 = []
    xs_atts2_2 = []
    xs_atts2_3 = []

    xs_atts3_1 = []
    xs_atts3_2 = []
    xs_atts3_3 = []

    xs_atts4_1 = []
    xs_atts4_2 = []
    xs_atts4_3 = []

    labels_x = []
    labels_y = []
    labels_z = []
    
    with torch.no_grad():
        for bi, inp in tqdm(enumerate(dataloader), desc="Cache"):
            batch_size = inp[f"base_tokens_{input_tp}"].size(0)

            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)

            with TraceDict(
                model,
                hook_points,
                retain_input=True,
            ) as cache:
                output = model(inp[f"base_tokens_{input_tp}"])
                
            for k, v in cache.items():
                if v is not None and isinstance(v, nethook.Trace):
                    cache[k].input = v.input.to("cpu")
                    cache[k].output = v.output[0].to("cpu")

            for	i in range(batch_size):
                pos_ents1 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["ents1"][i]).to("cpu")
                pos_ents2 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["ents2"][i]).to("cpu")
                pos_ents3 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["ents3"][i]).to("cpu")
                pos_atts1_1 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts1_1"][i]).to("cpu")
                pos_atts1_2 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts1_2"][i]).to("cpu")
                pos_atts1_3 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts1_3"][i]).to("cpu")
                pos_atts2_1 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts2_1"][i]).to("cpu")
                pos_atts2_2 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts2_2"][i]).to("cpu")
                pos_atts2_3 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts2_3"][i]).to("cpu")
                pos_atts3_1 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts3_1"][i]).to("cpu")
                pos_atts3_2 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts3_2"][i]).to("cpu")
                pos_atts3_3 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts3_3"][i]).to("cpu")
                pos_atts4_1 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts4_1"][i]).to("cpu")
                pos_atts4_2 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts4_2"][i]).to("cpu")
                pos_atts4_3 = compute_prev_clue_pos(inp[f"base_tokens_{input_tp}"][i], inp["atts4_3"][i]).to("cpu")

                if (torch.numel(pos_ents1) == 0 or torch.numel(pos_ents2) == 0 or torch.numel(pos_ents3) == 0
                    or torch.numel(pos_atts1_1) == 0 or torch.numel(pos_atts1_2) == 0 or torch.numel(pos_atts1_3) == 0
                    or torch.numel(pos_atts2_1) == 0 or torch.numel(pos_atts2_2) == 0 or torch.numel(pos_atts2_3) == 0
                    or torch.numel(pos_atts3_1) == 0 or torch.numel(pos_atts3_2) == 0 or torch.numel(pos_atts3_3) == 0
                    or torch.numel(pos_atts4_1) == 0 or torch.numel(pos_atts4_2) == 0 or torch.numel(pos_atts4_3) == 0):
                    continue
                
                xs_ents1.append(cache[f"model.layers.{layer}"].output[i][pos_ents1])
                xs_ents2.append(cache[f"model.layers.{layer}"].output[i][pos_ents2])
                xs_ents3.append(cache[f"model.layers.{layer}"].output[i][pos_ents3])

                xs_atts1_1.append(cache[f"model.layers.{layer}"].output[i][pos_atts1_1])
                xs_atts1_2.append(cache[f"model.layers.{layer}"].output[i][pos_atts1_2])
                xs_atts1_3.append(cache[f"model.layers.{layer}"].output[i][pos_atts1_3])

                xs_atts2_1.append(cache[f"model.layers.{layer}"].output[i][pos_atts2_1])
                xs_atts2_2.append(cache[f"model.layers.{layer}"].output[i][pos_atts2_2])
                xs_atts2_3.append(cache[f"model.layers.{layer}"].output[i][pos_atts2_3])

                xs_atts3_1.append(cache[f"model.layers.{layer}"].output[i][pos_atts3_1])
                xs_atts3_2.append(cache[f"model.layers.{layer}"].output[i][pos_atts3_2])
                xs_atts3_3.append(cache[f"model.layers.{layer}"].output[i][pos_atts3_3])

                xs_atts4_1.append(cache[f"model.layers.{layer}"].output[i][pos_atts4_1])
                xs_atts4_2.append(cache[f"model.layers.{layer}"].output[i][pos_atts4_2])
                xs_atts4_3.append(cache[f"model.layers.{layer}"].output[i][pos_atts4_3])

                labels_x.append(inp["labels_x"][i])
                labels_y.append(inp["labels_y"][i])
                labels_z.append(inp["labels_z"][i])
                
            del cache, inp
            torch.cuda.empty_cache()
            
    xs_ents1 = torch.cat(xs_ents1, 0).float()
    xs_ents2 = torch.cat(xs_ents2, 0).float()
    xs_ents3 = torch.cat(xs_ents3, 0).float()

    xs_atts1_1 = torch.cat(xs_atts1_1, 0).float()
    xs_atts1_2 = torch.cat(xs_atts1_2, 0).float()
    xs_atts1_3 = torch.cat(xs_atts1_3, 0).float()

    xs_atts2_1 = torch.cat(xs_atts2_1, 0).float()
    xs_atts2_2 = torch.cat(xs_atts2_2, 0).float()
    xs_atts2_3 = torch.cat(xs_atts2_3, 0).float()

    xs_atts3_1 = torch.cat(xs_atts3_1, 0).float()
    xs_atts3_2 = torch.cat(xs_atts3_2, 0).float()
    xs_atts3_3 = torch.cat(xs_atts3_3, 0).float()

    xs_atts4_1 = torch.cat(xs_atts4_1, 0).float()
    xs_atts4_2 = torch.cat(xs_atts4_2, 0).float()
    xs_atts4_3 = torch.cat(xs_atts4_3, 0).float()

    labels_x = torch.stack(labels_x)
    labels_y = torch.stack(labels_y)
    labels_z = torch.stack(labels_z)
    
    return (xs_ents1, xs_ents2, xs_ents3,
            xs_atts1_1, xs_atts1_2, xs_atts1_3,
            xs_atts2_1, xs_atts2_2, xs_atts2_3,
            xs_atts3_1, xs_atts3_2, xs_atts3_3,
            xs_atts4_1, xs_atts4_2, xs_atts4_3,
            labels_x, labels_y, labels_z,)


def extract_activation_for_visualization(
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        list_datafile: list,
        sfout: str,
        num_samples: int,
        layer: int=15,
        input_tp: str="table",
        batch_size: int=100,
        ):
    
    dataloader_tb = load_dataloader(tokenizer, list_datafile, num_samples, batch_size=batch_size,)
    (xs_ents1, xs_ents2, xs_ents3,
     xs_atts1_1, xs_atts1_2, xs_atts1_3,
     xs_atts2_1, xs_atts2_2, xs_atts2_3,
     xs_atts3_1, xs_atts3_2, xs_atts3_3,
     xs_atts4_1, xs_atts4_2, xs_atts4_3,
     labels_x, labels_y, labels_z, ) = get_x_y(model, dataloader_tb, layer, input_tp)

    data_out = {
        "ent1": xs_ents1,
        "ent2": xs_ents2,
        "ent3": xs_ents3,
        "att1_1": xs_atts1_1,
        "att1_2": xs_atts1_2,
        "att1_3": xs_atts1_3,
        "att2_1": xs_atts2_1,
        "att2_2": xs_atts2_2,
        "att2_3": xs_atts2_3,
        "att3_1": xs_atts3_1,
        "att3_2": xs_atts3_2,
        "att3_3": xs_atts3_3,
        "att4_1": xs_atts4_1,
        "att4_2": xs_atts4_2,
        "att4_3": xs_atts4_3,
        "xi": labels_x,
        "yi": labels_y,
        "zi": labels_z,
        }

    torch.save(data_out, sfout)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for activation patching")
    
    parser.add_argument("--llm_tp", type=str, default="llama", help="llama / qwen")
    parser.add_argument("--input_tp", type=str, default="story", help="table / temp / story")
    parser.add_argument("--layer", type=int, default=15, help="")
    parser.add_argument("--num_samples", type=int, default=300, help="")
    parser.add_argument("--batch_size", type=int, default=100, help="")
    parser.add_argument("--sd_in", type=str, default="./data/", help="")
    parser.add_argument("--sd_out", type=str, default="./data_emb/", help="")
    args = parser.parse_args()
    
    ###Extract Activation for Visualization###
    llm_tp = args.llm_tp
    dv_id = args.input_tp
    layer = args.layer
    
    device = torch.device(f"cuda:{dv_id}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(llm_tp, device)
    print("Model and Tokenizer loaded")
    sd_in = args.sd_in
    sd_out = args.sd_out

    num_samples = args.num_samples
    batch_size = args.batch_size
    
    for data_tp in ["city", "create", "job", "relation", "space"]:
        datafile = f"./data/{data_tp}_tss_all.jsonl"
        for input_tp in ["table", "temp", "story",]:
                sfout = sd_out + f"{llm_tp}_l{layer}_{data_tp}_{input_tp}.pt"
                print(f"Processing {data_tp} dataset {input_tp} input ...")
                extract_activation_for_visualization(
                    tokenizer,
                    model,
                    datafile,
                    sfout,
                    num_samples,
                    layer=int(layer),
                    input_tp=input_tp,
                    batch_size=batch_size,
                )

    ###
    
