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


def load_tb_data(tokenizer, num_samples, lst_data_file, rand=True, vari="p_cr", ablate_t=False):
    """
    vari in ["p_cr", "r_cr"]
    """
    data = []
    for data_file in lst_data_file:
        with open(data_file, encoding="utf-8") as f:
            data_ = [json.loads(line) for line in f]
            data.extend(data_)

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
    labels_z_r_cr = []
    labels_z_p_cr = []
    
    prompts_temp = []
    prompts_table = []
    prompts_story = []

    prompts_temp_cr = []
    prompts_table_cr = []
    prompts_story_cr = []
    
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
        ##table ablation##
        if ablate_t:
            if random.random() < 0.5:
                ctx_t = ablate_table(ctx_t, att3_2)
        ##
        prompts_table.append(ctx_t)
        prompts_temp.append(ctx)
        prompts_story.append(ctx_s)
        
        ctx = "Context: " + data[i][f"input_{vari}"].strip()
        ctx_t = "Context: " + data[i][f"t_input_{vari}"].strip()
        ctx_s = "Context: " + data[i][f"s_input_{vari}"].strip()
        ##table ablation## 
        if ablate_t:
            if random.random() < 0.5:
                ctx_t = ablate_table(ctx_t, att3_2)
        ##
        prompts_table_cr.append(ctx_t)
        prompts_temp_cr.append(ctx)
        prompts_story_cr.append(ctx_s)
        
        ##label info.
        t_lws = data[i]["t_input"].split("|")#e.g., ['', ' Entity ', ' Left ', ' Right ', ' Front ', ' Back ', ...]
        t_lws_r_cr = data[i]["t_input_r_cr"].split("|")
        all_tps = [t_lws[1].strip(), t_lws[2].strip(), t_lws[3].strip(), t_lws[4].strip(), t_lws[5].strip(),
                   t_lws_r_cr[2].strip(), t_lws_r_cr[3].strip(), t_lws_r_cr[4].strip(), t_lws_r_cr[5].strip()]
        t_lws_p_cr = data[i]["t_input_p_cr"].split("|")
        tps_p_cr = [t_lws_p_cr[1].strip(), t_lws_p_cr[2].strip(), t_lws_p_cr[3].strip(), t_lws_p_cr[4].strip(), t_lws_p_cr[5].strip()]

        label_y = [1, 2, 3,]
        label_x = [1, 2, 3, 4, 5]
        label_z = [1, 2, 3, 4, 5]
        label_z_r_cr = [1, 6, 7, 8, 9]
        label_z_p_cr = [1,]
        for tp in tps_p_cr[1:]:
            label_z_p_cr.append(all_tps.index(tp) + 1)

        labels_y.append(label_y)
        labels_x.append(label_x)
        labels_z.append(label_x)
        labels_z_r_cr.append(label_z_r_cr)
        labels_z_p_cr.append(label_z_p_cr)
        ##
        
    input_tokens_table = tokenizer(prompts_table, padding=True, return_tensors="pt")
    input_ids_table = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp, padding=True, return_tensors="pt")
    input_ids_temp = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story, padding=True, return_tensors="pt")
    input_ids_story = input_tokens_story["input_ids"]

    input_tokens_table_cr = tokenizer(prompts_table_cr, padding=True, return_tensors="pt")
    input_ids_table_cr = input_tokens_table_cr["input_ids"]
    input_tokens_temp_cr = tokenizer(prompts_temp_cr, padding=True, return_tensors="pt")
    input_ids_temp_cr = input_tokens_temp_cr["input_ids"]
    input_tokens_story_cr = tokenizer(prompts_story_cr, padding=True, return_tensors="pt")
    input_ids_story_cr = input_tokens_story_cr["input_ids"]
    
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
    labels_z_r_cr = torch.tensor(labels_z_r_cr)
    labels_z_p_cr = torch.tensor(labels_z_p_cr)
    
    if vari == "p_cr":
        labels_z_cr = labels_z_p_cr
    elif vari == "r_cr":
        labels_z_cr = labels_z_r_cr
    
    return (input_ids_table, input_ids_temp, input_ids_story,
            ents1, ents2, ents3,
            atts1_1, atts1_2, atts1_3,
            atts2_1, atts2_2, atts2_3,
            atts3_1, atts3_2, atts3_3,
            atts4_1, atts4_2, atts4_3,
            input_ids_table_cr, input_ids_temp_cr, input_ids_story_cr,
            labels_x, labels_y, labels_z, labels_z_cr,)


def load_dataloader(
        tokenizer: AutoTokenizer,
        lst_data_file: list,
        num_samples: int,
        batch_size: int,
        vari: str="p_cr",
        ablate_t: bool=False,):
    
    raw_data = load_tb_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        lst_data_file=lst_data_file,
        vari=vari,
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

    base_tokens_table_cr = raw_data[18]
    base_tokens_temp_cr = raw_data[19]
    base_tokens_story_cr = raw_data[20]

    labels_x = raw_data[21]
    labels_y = raw_data[22]
    labels_z = raw_data[23]
    labels_z_cr = raw_data[24]

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
            "base_tokens_table_cr": base_tokens_table_cr,
            "base_tokens_temp_cr": base_tokens_temp_cr,
            "base_tokens_story_cr": base_tokens_story_cr,
            "labels_x": labels_x,
            "labels_y":	labels_y,
            "labels_z":	labels_z,
            "labels_z_cr": labels_z_cr,
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
    input_tp: "table", "table_cr", "temp", "temp_cr", "story", "story_cr"

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
    labels_z_cr = []
    
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
                labels_z_cr.append(inp["labels_z_cr"][1])
                
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
    labels_z_cr = torch.stack(labels_z_cr)
    
    return (xs_ents1, xs_ents2, xs_ents3,
            xs_atts1_1, xs_atts1_2, xs_atts1_3,
            xs_atts2_1, xs_atts2_2, xs_atts2_3,
            xs_atts3_1, xs_atts3_2, xs_atts3_3,
            xs_atts4_1, xs_atts4_2, xs_atts4_3,
            labels_x, labels_y, labels_z, labels_z_cr,)


def extract_activation_for_visualization(
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        list_datafile: list,
        sfout: str,
        num_samples: int,
        layer: int=15,
        input_tp: str="table",
        batch_size: int=100,
        vari: str="r_cr",
        ):
    
    dataloader_tb = load_dataloader(tokenizer, list_datafile, num_samples, batch_size=batch_size, vari=vari)
    (xs_ents1, xs_ents2, xs_ents3,
     xs_atts1_1, xs_atts1_2, xs_atts1_3,
     xs_atts2_1, xs_atts2_2, xs_atts2_3,
     xs_atts3_1, xs_atts3_2, xs_atts3_3,
     xs_atts4_1, xs_atts4_2, xs_atts4_3,
     labels_x, labels_y, labels_z, labels_z_cr,) = get_x_y(model, dataloader_tb, layer, input_tp)

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
        "zi_cr": labels_z_cr,
        }

    torch.save(data_out, sfout)


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result
    

class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias
    
    
class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False, normalize: bool = False) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

        
    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] | None = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)
        
        return latents_pre_act, latents, recons


class AELinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_features: int=1024, 
                 device=None):
        super().__init__()
        self.out_features = out_features
        self.mid_features = mid_features
        self.in_features = in_features
        self.device = device
        
        self.relu = nn.ReLU()
        self.linear_encoder = nn.Linear(in_features, mid_features)
        self.encoder = nn.Sequential(
            #self.relu,
            self.linear_encoder,
            #self.relu,
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(mid_features, out_features)
        ).to(device)
        
    def topk(self, x, k=7):
        # Get top-k values
        topk_vals, topk_idx = torch.topk(x, k, dim=-1)
        # Zero out everything else
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
        return x * mask

    def forward(self, x):
        lat_x = self.encoder(x)
        #reconstructed = self.decoder(lat_x)
        #reconstructed = self.topk(lat_x)
        reconstructed = lat_x
        return reconstructed
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder


class ProbeDataset(Dataset):
    def __init__(self, X_ctx, y_xi, y_yi, y_zi, y_zi_cr):
        self.features_ctx = X_ctx
        self.y_xi = y_xi
        self.y_yi = y_yi
        self.y_zi = y_zi
        self.y_zi_cr = y_zi_cr
    
    def __getitem__(self, index):
        one_x_ctx = self.features_ctx[index].to(device)
        one_y_xi = self.y_xi[index].to(device)
        one_y_yi = self.y_yi[index].to(device)
        one_y_zi = self.y_zi[index].to(device)
        one_y_zi_cr = self.y_zi_cr[index].to(device)
        return one_x_ctx, one_y_xi, one_y_yi, one_y_zi, one_y_zi_cr
    
    def __len__(self):
        return self.features_ctx.shape[0]


def cal_acc(model, dev_loader, y_tp="x"):
    model.eval()
    correct_ctx = 0.0
    total_examples = 0
    for features_ctx, labels_x, labels_y, labels_z, labels_z_cr in dev_loader:
        if y_tp == "x":
            labels = labels_x
        elif y_tp == "y":
            labels = labels_y
        elif y_tp == "z":
            labels = labels_z
        elif y_tp == "z_cr":
            labels = labels_z_cr
        with torch.no_grad():
            logits_ctx = model(features_ctx)
        pred_ctx = logits_ctx.argmax(dim=1)

        correct_ctx += (labels == pred_ctx).sum().item()
        total_examples += features_ctx.size(0)
    acc_ctx = correct_ctx / total_examples
    return acc_ctx


def train_probe_model(model, train_loader, dev_loader, 
                      num_epochs=100, lr=0.02, lr_each=20, y_tp="x"):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    max_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features_ctx, labels_x, labels_y, labels_z, labels_z_cr) in enumerate(train_loader):
            if y_tp == "x":
                labels = labels_x
            elif y_tp == "y":
                labels = labels_y
            elif y_tp == "z":
                labels = labels_z
            elif y_tp == "z_cr":
                labels = labels_z_cr
              
            logits_ctx = model(features_ctx)
            loss = criterion(logits_ctx, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            acc_ctx = cal_acc(model, dev_loader, y_tp=y_tp)
            acc = acc_ctx
            if acc > max_acc:
                print(f"Acc. Ctx.: {acc_ctx:.3f}.")
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}"
                  f" | Val Acc.: {acc:.3f}")
            
        ###Adjust lr###
        lrn = int(num_epochs / lr_each)
        if (epoch + 1) % lr_each == 0:
            lri = (epoch + 1) / lr_each
            for group in optimizer.param_groups:
                group['lr'] = lr - lr * (1 / lrn) * lri


def train_probe_main(model, tokenizer, ae_probe, list_datafile, num_samples=100, batch_size=10,
                     input_tp="table", vari="p_cr", y_tp="x", ablate_t=False, num_epochs=200, lr=0.03):
    dataloader_tb = load_dataloader(tokenizer, list_datafile, num_samples=num_samples, batch_size=batch_size, vari=vari, ablate_t=ablate_t)
    (xs_ents1, xs_ents2, xs_ents3,
     xs_atts1_1, xs_atts1_2, xs_atts1_3,
     xs_atts2_1, xs_atts2_2, xs_atts2_3,
     xs_atts3_1, xs_atts3_2, xs_atts3_3,
     xs_atts4_1, xs_atts4_2, xs_atts4_3,
     labels_x, labels_y, labels_z, labels_z_cr,) = get_x_y(model, dataloader_tb, layer, input_tp=input_tp)
    
    y_xi_1 = labels_x[:, 1]
    y_xi_2 = labels_x[:, 2]
    y_xi_3 = labels_x[:, 3]
    y_xi_4 = labels_x[:, 4]
    
    y_yi_1 = labels_y[:, 0]
    y_yi_2 = labels_y[:, 1]
    y_yi_3 = labels_y[:, 2]

    y_zi_1 = labels_z[:, 1]
    y_zi_2 = labels_z[:, 2]
    y_zi_3 = labels_z[:, 3]
    y_zi_4 = labels_z[:, 4]

    y_zi_cr_1 = labels_z_cr[:, 1]
    y_zi_cr_2 = labels_z_cr[:, 2]
    y_zi_cr_3 = labels_z_cr[:, 3]
    y_zi_cr_4 = labels_z_cr[:, 4]

    X_ctx = torch.cat([xs_atts1_1, xs_atts1_2, xs_atts1_3,
                       xs_atts2_1, xs_atts2_2, xs_atts2_3,
                       xs_atts3_1, xs_atts3_2, xs_atts3_3,
                       xs_atts4_1, xs_atts4_2, xs_atts4_3,], dim=0)

    y_xi = torch.cat((y_xi_1, y_xi_1, y_xi_1,
                      y_xi_2, y_xi_2, y_xi_2,
                      y_xi_3, y_xi_3, y_xi_3,
                      y_xi_4, y_xi_4, y_xi_4,), dim=0)

    y_yi = torch.cat((y_yi_1, y_yi_2, y_xi_3,
                      y_yi_1, y_yi_2, y_xi_3,
                      y_yi_1, y_yi_2, y_xi_3,
                      y_yi_1, y_yi_2, y_xi_3,),	dim=0)

    y_zi = torch.cat((y_zi_1, y_zi_1, y_zi_1,
                      y_zi_2, y_zi_2, y_zi_2,
                      y_zi_3, y_zi_3, y_zi_3,
                      y_zi_4, y_zi_4, y_zi_4,),	dim=0)

    y_zi_cr = torch.cat((y_zi_cr_1, y_zi_cr_1, y_zi_cr_1,
                         y_zi_cr_2, y_zi_cr_2, y_zi_cr_2,
                         y_zi_cr_3, y_zi_cr_3, y_zi_cr_3,
                         y_zi_cr_4, y_zi_cr_4, y_zi_cr_4,), dim=0)
    
    (X_ctx_train, X_ctx_test,
     y_xi_train, y_xi_test,
     y_yi_train, y_yi_test,
     y_zi_train, y_zi_test,
     y_zi_cr_train, y_zi_cr_test,) = train_test_split(X_ctx, y_xi, y_yi, y_zi, y_zi_cr, test_size=0.5, random_state=42)
    
    train_ds = ProbeDataset(X_ctx_train, y_xi_train, y_yi_train, y_zi_train, y_zi_cr_train)
    test_ds = ProbeDataset(X_ctx_test, y_xi_test, y_yi_test, y_zi_test, y_zi_cr_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=20,
        shuffle=True,
        num_workers=0,)

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=20,
        shuffle=True,
        num_workers=0,)

    torch.manual_seed(123)
    train_probe_model(ae_probe, train_loader, test_loader, num_epochs=num_epochs, lr=lr, y_tp=y_tp)
    
    
if __name__ == "__main__":
    """
    llm_tp = sys.argv[1]
    dv_id = sys.argv[2]
    layer = sys.argv[3]
    y_tp = sys.argv[4]
    data_tp = sys.argv[5]
    input_tp = sys.argv[6]

    num_samples = 200
    batch_size = 30
    lr = 0.008
    num_epochs = 100
    device = torch.device(f"cuda:{dv_id}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(llm_tp, device)
    print("Model and Tokenizer loaded")
    in_features = model.config.hidden_size
    sd_in = "/work01/daiqin/activation_patching/poe_project_copy/data/data_table/"
    list_datafile = [sd_in + f"{data_tp}_tts_0.jsonl", sd_in + f"{data_tp}_tts_1.jsonl", sd_in + f"{data_tp}_tts_2.jsonl",  sd_in + f"{data_tp}_tts_4.jsonl",
                     sd_in + f"{data_tp}_tts_5.jsonl", sd_in + f"{data_tp}_tts_6.jsonl", sd_in + f"{data_tp}_tts_7.jsonl",  sd_in + f"{data_tp}_tts_8.jsonl",
                     sd_in + f"{data_tp}_tts_9.jsonl", sd_in + f"{data_tp}_tts_10.jsonl", sd_in + f"{data_tp}_tts_11.jsonl",  sd_in + f"{data_tp}_tts_12.jsonl",]
    
    if y_tp == "x":
        ae_probe = AELinear(in_features=in_features, mid_features=256, out_features=5)
        ae_probe = ae_probe.to(device)
        train_probe_main(model, tokenizer, ae_probe, list_datafile, num_samples=num_samples, batch_size=batch_size,
                         input_tp=input_tp, vari="p_cr", y_tp=y_tp,  num_epochs=100)
        
    elif y_tp == "y":
        ae_probe = AELinear(in_features=in_features, mid_features=256, out_features=3)
        ae_probe = ae_probe.to(device)
        train_probe_main(model, tokenizer, ae_probe, list_datafile, num_samples=num_samples, batch_size=batch_size,
                         input_tp=input_tp, vari="p_cr", y_tp=y_tp, num_epochs=100)

    elif y_tp == "z":
        ae_probe = AELinear(in_features=in_features, mid_features=256, out_features=9)
        ae_probe = ae_probe.to(device)
        train_probe_main(model, tokenizer, ae_probe, list_datafile, num_samples=num_samples, batch_size=batch_size,
                         input_tp=input_tp, vari="r_cr", y_tp=y_tp, num_epochs=num_epochs, lr=lr, ablate_t=True)
        train_probe_main(model, tokenizer, ae_probe, list_datafile, num_samples=num_samples, batch_size=batch_size,
                         input_tp=f"{input_tp}_cr", vari="p_cr", y_tp="z_cr", num_epochs=num_epochs, lr=lr, ablate_t=True)
        #train_probe_main(model, tokenizer, ae_probe, list_datafile, num_samples=num_samples, batch_size=batch_size,
        #                 input_tp=f"{input_tp}_cr", vari="r_cr", y_tp="z_cr", num_epochs=num_epochs, lr=lr)
    

    """
    ###Extract Activation for Visualization###
    llm_tp = sys.argv[1]
    dv_id = sys.argv[2]
    layer = sys.argv[3]
    
    device = torch.device(f"cuda:{dv_id}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(llm_tp, device)
    print("Model and Tokenizer loaded")
    sd_in = "/work01/daiqin/activation_patching/poe_project_copy/data/data_table/"
    sd_out = "/work01/daiqin/activation_patching/poe_project_copy/data/data_table_emb/"

    num_samples = 300
    batch_size = 100
    
    for data_tp in ["city", "create", "job", "relation", "space"]:
        for input_tp in ["table", "temp", "story", "table_cr", "temp_cr", "story_cr"]:
            for vari in ["p_cr", "r_cr"]:
                print(f"Processing {data_tp} dataset {input_tp} input ...")
                list_datafile = [sd_in + f"{data_tp}_tts_0.jsonl", sd_in + f"{data_tp}_tts_1.jsonl", sd_in + f"{data_tp}_tts_2.jsonl",  sd_in + f"{data_tp}_tts_4.jsonl",
                                 sd_in + f"{data_tp}_tts_5.jsonl", sd_in + f"{data_tp}_tts_6.jsonl", sd_in + f"{data_tp}_tts_7.jsonl",  sd_in + f"{data_tp}_tts_8.jsonl",
                                 sd_in + f"{data_tp}_tts_9.jsonl", ]
                sfout = sd_out + f"{llm_tp}_l{layer}_{data_tp}_{input_tp}_{vari}.pt"
                sfout_all = sd_in + f"{data_tp}_tts_all.jsonl"
                combine_data(list_datafile, sfout_all)
                
                extract_activation_for_visualization(
                    tokenizer,
                    model,
                    list_datafile,
                    sfout,
                    num_samples,
                    layer=int(layer),
                    input_tp=input_tp,
                    batch_size=batch_size,
                    vari=vari,
                )

    ###
    
