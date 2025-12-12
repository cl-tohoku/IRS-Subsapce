import random
import fire
import numpy as np
from functools import partial
from baukit import TraceDict
from tqdm import tqdm
import json
import sys
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import transformers
from torch.utils.data import DataLoader
from datasets import Dataset as Dataset_ds
from baukit import TraceDict, nethook
from einops import rearrange, einsum

import argparse

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
        else:
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
        #model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,)
    return model, tokenizer


def gen_que_temp(data_tp="space", input_tp="story", atti=2):
    prefix = "Based on the context like"
    if input_tp == "table":
        prefix = "Thus in the table"
    ENT = "{ENT}"
    if data_tp == "space":
        if atti == 2:
            if input_tp == "table":
                que_temp = f" {prefix} the entity is banned in {ENT} and designed in"
            else:
                que_temp = f" {prefix} the entity that is banned in {ENT} and designed in"
            att_tp = "atts2"
        elif atti == 3:
            if input_tp == "table":
                que_temp = f" {prefix} the entity is banned in {ENT} and exported to"
            else:
                que_temp = f" {prefix} the entity that is banned in {ENT} and exported to"
            att_tp = "atts3"
        elif atti == 1:
            if input_tp == "table":
                que_temp = f" {prefix} the entity is banned in {ENT} and manufactured in"
            else:
                que_temp = f" {prefix} the entity that is banned in {ENT} and produced in"
            att_tp = "atts1"
    elif data_tp == "create":
        if atti == 2:
            if input_tp == "table":
                que_temp = f" {prefix} the person's favorite object is the {ENT} and his bought object is the" 
            else:
                que_temp = f" {prefix} the person's favorite object is the {ENT} and he bought the" 
            att_tp = "atts2"
        elif atti == 3:
            if input_tp == "table":
                que_temp = f" {prefix} the person's favorite object is the {ENT} and sold object is the"
            else:
                que_temp = f" {prefix} the person's favorite object is the {ENT} and he sold the"
            att_tp = "atts3"
        elif atti == 1:
            if input_tp == "table":
                que_temp = f" {prefix} the person's fovarite object is {ENT} and his created object is"
            else:
                que_temp = f" {prefix} the person's fovarite object is the {ENT} and he created the"
            att_tp = "atts1"
    elif data_tp == "job":
        if atti == 2:
            if input_tp == "table":
                que_temp = f" {prefix} the person's disliked job is {ENT} and his dream job is"
            else:
                que_temp = f" {prefix} the person dislikes being a {ENT} and dreams to be a"
            att_tp = "atts2"
        elif atti == 3:
            if input_tp == "table":
                que_temp = f" {prefix} the person's disliked job is {ENT} and his previous job is"
            else:
                que_temp = f" {prefix} the person dislikes being a {ENT} and his previous job was a"
            att_tp = "atts3"
        elif atti == 1:
            if input_tp == "table":
                que_temp = f" {prefix} the person's disliked job is {ENT} and his current job is"
            else:
                que_temp = f" {prefix} the person dislikes to be a {ENT} and now he is a"
            att_tp = "atts1"
    elif data_tp == "relation":
        if atti == 2:
            if input_tp == "table":
                que_temp = f" {prefix} the person's boss is {ENT} and his child is"
            else:
                que_temp = f" {prefix} the person works under {ENT} and has a child named"
            att_tp = "atts2"
        elif atti == 3:
            if input_tp == "table":
                que_temp = f" {prefix} the person's boss is {ENT} and his teacher is"
            else:
                que_temp = f" {prefix} the person works under {ENT} and has a teacher named"
            att_tp = "atts3"
        elif atti == 1:
            if input_tp == "table":
                que_temp = f" {prefix} the person's boss is {ENT} and his spouse is"
            else:
                que_temp = f" {prefix} the person, who works under {ENT}, married to"
            att_tp = "atts1"
    elif data_tp == "city":
        if atti == 2:
            if input_tp == "table":
                que_temp = f" {prefix} the person's disliked city is {ENT} and his Lived City is"
            else:
                que_temp = f" {prefix} the person dislikes {ENT} and he lives and resides in"
            att_tp = "atts2"
        elif atti == 3:
            if input_tp == "table":
                que_temp = f" {prefix} the person's disliked city is {ENT} and his loved city is"
            else:
                que_temp = f" {prefix} the person dislikes {ENT} but he likes"
            att_tp = "atts3"
        elif atti == 1:
            if input_tp == "table":
                que_temp = f" {prefix} the person's disliked city is {ENT} and his birthplace is"
            else:
                que_temp = f" {prefix} the person dislikes {ENT} and he is from"
            att_tp = "atts1"
    return que_temp, att_tp


def gen_que_temp_ent(data_tp="space"):
    prefix = "Thus in the table"
    ATT = "{ATT}"
    if data_tp == "space":
        que_temp = f" {prefix} {ATT} designs the"
        att_tp = "atts2"
    elif data_tp == "create":
        que_temp = f" {prefix} the {ATT} is bought by"
        att_tp = "atts2"
    elif data_tp == "job":
        que_temp = f" {prefix} the job of being a {ATT} is dreamed by"
        att_tp = "atts2"
    elif data_tp == "relation":
        que_temp = f" {prefix} {ATT} is a child of"
        att_tp = "atts2"
    elif data_tp == "city":
        que_temp = f" {prefix} {ATT} is loved by"
        att_tp = "atts4"
    return que_temp, att_tp


def load_tb_data(tokenizer, num_samples, data_file, rand=True,
                 vari="p_cr", data_tp="space", input_tp="story", enti=1, atti=2):
    """
    vari in ["p_cr", "r_cr"]
    """
    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if rand:
        random.shuffle(data)

    #que_temp, que_att_tp = gen_que_temp(data_tp=data_tp, atti=atti, input_tp=input_tp)
    #que_temp, que_att_tp = gen_que_temp_ent(data_tp=data_tp)
    que_temp = "Based on the context, given like {ENT1} to {ATT1} , {ENT2} to"
    if enti == 0:
        enti_cr1 = 2
    elif enti == 1:
        enti_cr1 = 0
    elif enti == 2:
        enti_cr1 = 1
        
    que_att_tp = f"atts{atti}"
    
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

    labels = []
    labels1 = []
    labels2 = []
    labels3 = []
    
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
        
        ctx = "Context: " + data[i]["input"]
        ctx_t = "Context: " + data[i]["t_input"]
        ctx_s = "Context: " + data[i]["s_input"]
        ctx_t = ctx_t.replace("||", "|\n|")

        que_ent1 = data[i]["ents"][enti_cr1]#few shot
        que_ent2 = data[i]["ents"][enti]
        que_att1 = data[i][que_att_tp][enti_cr1]#few shot
        que_att2 = data[i][que_att_tp][enti]
        # e.g., "Based on the context, like fork ; Japan, flower ;"
        que = que_temp.replace("{ENT1}", que_ent1).replace("{ATT1}", que_att1).replace("{ENT2}", que_ent2)
        prompts_table.append(ctx_t + que)
        prompts_temp.append(ctx + que)
        prompts_story.append(ctx_s + que)
        labels.append(tokenizer.encode(" %s"%que_att2)[-1])
        
        labels1.append(tokenizer.encode(" %s"%data[i]["atts1"][enti])[-1])
        labels2.append(tokenizer.encode(" %s"%data[i]["atts2"][enti])[-1])
        labels3.append(tokenizer.encode(" %s"%data[i]["atts3"][enti])[-1])
        
        ctx = "Context: " + data[i][f"input_{vari}"]
        ctx_t = "Context: " + data[i][f"t_input_{vari}"]
        ctx_s = "Context: " + data[i][f"s_input_{vari}"]
        ctx_t =	ctx_t.replace("||", "|\n|")
        
        prompts_table_cr.append(ctx_t + que)
        prompts_temp_cr.append(ctx + que)
        prompts_story_cr.append(ctx_s + que)
        
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

    labels = torch.tensor(labels)
    labels1 = torch.tensor(labels1)
    labels2 = torch.tensor(labels2)
    labels3 = torch.tensor(labels3)
    
    return (input_ids_table, input_ids_temp, input_ids_story,
            ents1, ents2, ents3,
            atts1_1, atts1_2, atts1_3,
            atts2_1, atts2_2, atts2_3,
            atts3_1, atts3_2, atts3_3,
            atts4_1, atts4_2, atts4_3,
            input_ids_table_cr, input_ids_temp_cr, input_ids_story_cr,
            labels, labels1, labels2, labels3,)


def load_dataloader(
        tokenizer: AutoTokenizer,
        data_file: list,
        num_samples: int,
        batch_size: int,
        vari: str="p_cr",
        data_tp: str="space",
        input_tp: str="story",
        enti: int=2,
        atti: int=2,):
    
    raw_data = load_tb_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=data_file,
        vari=vari,
        data_tp=data_tp,
        input_tp=input_tp,
        enti=enti,
        atti=atti,
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

    labels = raw_data[21]
    labels1 = raw_data[22]
    labels2 = raw_data[23]
    labels3 = raw_data[24]

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
            "labels": labels,
            "labels1": labels1,
            "labels2": labels2,
            "labels3": labels3,
        }
    ).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_caches_resid_ap(
        model: AutoModelForCausalLM,
        dataloader: torch.utils.data.DataLoader,
        layer1=10,
        layer2=20,
        input_tp="table",):
    
    hook_points = [
        f"model.layers.{layer}"
        for layer in range(layer1, layer2)
    ]
    
    apply_softmax = torch.nn.Softmax(dim=-1)
    clean_cache, corrupt_cache = {}, {}
    clean_logit_outputs = defaultdict(dict)
    
    with torch.no_grad():
        for bi, inp in tqdm(enumerate(dataloader), desc="Clean cache"):
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

            clean_cache[bi] = cache
            del output, cache, inp
            torch.cuda.empty_cache()
        print("CLEAN CACHE COMPUTED")

        for bi, inp in tqdm(enumerate(dataloader), desc="Corrupt cache"):
            for k, v in inp.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inp[k] = v.to(model.device)
            with TraceDict(
                model,
                hook_points,
                retain_input=True,
            ) as cache:
                output = model(inp[f"base_tokens_{input_tp}_cr"])

            for k, v in cache.items():
                if v is not None and isinstance(v, nethook.Trace):
                    cache[k].input = v.input.to("cpu")
                    cache[k].output = v.output[0].to("cpu")

            corrupt_cache[bi] = cache
            del output, cache, inp
            torch.cuda.empty_cache()

        print("CORRUPT CACHE COMPUTED")

    return (
        clean_cache,
        corrupt_cache,
        hook_points,
    )


def activation_patching_resid_ap(
        inputs: tuple=None,
        output: torch.Tensor=None,
        layer: str=None,
        model: AutoModelForCausalLM=None,
        source_cache: dict=None,
        bi: int=None,
        input_tokens: list=None,
        input_tp: str="table",
        proj_ma: torch.tensor=None,
        use_rand_proj_ma: int=0,
        patch_tp: str="modify",
        beta: float=0.55,
        ):
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(output, tuple):
        outputs = output[0]

    if use_rand_proj_ma == 1:
        low, high = proj_ma.min(), proj_ma.max()
        proj_ma_rand = (high - low) * torch.rand_like(proj_ma) + low
        
    outputs = rearrange(
        outputs,
        "batch seq_len d_resid -> batch seq_len d_resid",
        d_resid=model.config.hidden_size,
    )
    
    cache = rearrange(
        source_cache[bi][layer].output,
        "batch seq_len d_resid -> batch seq_len d_resid",
        d_resid=model.config.hidden_size,
    ).to(model.device)
    
    layer_index = int(layer.split(".")[2])
    
    for batch in range(outputs.size(0)):
        nb_tok = outputs.size(1)
        pos11 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts1_1"][batch],
        )
        pos12 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts1_2"][batch],
        )
        pos13 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts1_3"][batch],
        )
        pos21 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts2_1"][batch],
        )
        pos22 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts2_2"][batch],
        )
        pos23 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts2_3"][batch],
        )
        pos31 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts3_1"][batch],
        )
        pos32 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts3_2"][batch],
        )
        pos33 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts3_3"][batch],
        )
        pos41 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts4_1"][batch],
        )
        pos42 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts4_2"][batch],
        )
        pos43 = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}"][batch],
            input_tokens["atts4_3"][batch],
        )

        pos11_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts1_1"][batch],
        )
        pos12_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts1_2"][batch],
        )
        pos13_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts1_3"][batch],
        )
        pos21_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts2_1"][batch],
        )
        pos22_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts2_2"][batch],
        )
        pos23_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts2_3"][batch],
        )
        pos31_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts3_1"][batch],
        )
        pos32_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts3_2"][batch],
        )
        pos33_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts3_3"][batch],
        )
        pos41_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts4_1"][batch],
        )
        pos42_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts4_2"][batch],
        )
        pos43_cr = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}_cr"][batch],
            input_tokens["atts4_3"][batch],
        )
        
        try:
            if patch_tp=="replace":
                emb11_cr = cache[batch, pos11_cr]
                emb12_cr = cache[batch, pos12_cr]
                emb13_cr = cache[batch, pos13_cr]
                emb21_cr = cache[batch, pos21_cr]
                emb22_cr = cache[batch, pos22_cr]
                emb23_cr = cache[batch, pos23_cr]
                emb31_cr = cache[batch, pos31_cr]
                emb32_cr = cache[batch, pos32_cr]
                emb33_cr = cache[batch, pos33_cr]
                emb41_cr = cache[batch, pos41_cr]
                emb42_cr = cache[batch, pos42_cr]
                emb43_cr = cache[batch, pos43_cr]
            elif patch_tp=="modify":
                emb11_cr = outputs[batch, pos11]
                emb12_cr = outputs[batch, pos12]
                emb13_cr = outputs[batch, pos13]
                emb21_cr = outputs[batch, pos21]
                emb22_cr = outputs[batch, pos22]
                emb23_cr = outputs[batch, pos23]
                emb31_cr = outputs[batch, pos31]
                emb32_cr = outputs[batch, pos32]
                emb33_cr = outputs[batch, pos33]
                emb41_cr = outputs[batch, pos41]
                emb42_cr = outputs[batch, pos42]
                emb43_cr = outputs[batch, pos43]
            
                p_emb11_cr = torch.matmul(emb11_cr, proj_ma)
                p_emb12_cr = torch.matmul(emb12_cr, proj_ma)
                p_emb13_cr = torch.matmul(emb13_cr, proj_ma)
                p_emb21_cr = torch.matmul(emb21_cr, proj_ma)
                p_emb22_cr = torch.matmul(emb22_cr, proj_ma)
                p_emb23_cr = torch.matmul(emb23_cr, proj_ma)
                p_emb31_cr = torch.matmul(emb31_cr, proj_ma)
                p_emb32_cr = torch.matmul(emb32_cr, proj_ma)
                p_emb33_cr = torch.matmul(emb33_cr, proj_ma)
                p_emb41_cr = torch.matmul(emb41_cr, proj_ma)
                p_emb42_cr = torch.matmul(emb42_cr, proj_ma)
                p_emb43_cr = torch.matmul(emb43_cr, proj_ma)
            
                if use_rand_proj_ma == 1:
                    proj_ma = proj_ma_rand

                emb11_cr = emb11_cr + beta * torch.matmul(p_emb11_cr, proj_ma.T)
                emb12_cr = emb12_cr + beta * torch.matmul(p_emb12_cr, proj_ma.T)
                emb13_cr = emb13_cr + beta * torch.matmul(p_emb13_cr, proj_ma.T)
                emb21_cr = emb21_cr + beta * torch.matmul(p_emb21_cr, proj_ma.T)
                emb22_cr = emb22_cr + beta * torch.matmul(p_emb22_cr, proj_ma.T)
                emb23_cr = emb23_cr + beta * torch.matmul(p_emb23_cr, proj_ma.T)
                emb31_cr = emb31_cr + beta * torch.matmul(p_emb31_cr, proj_ma.T)
                emb32_cr = emb32_cr + beta * torch.matmul(p_emb32_cr, proj_ma.T)
                emb33_cr = emb33_cr + beta * torch.matmul(p_emb33_cr, proj_ma.T)
                emb41_cr = emb41_cr + beta * torch.matmul(p_emb41_cr, proj_ma.T)
                emb42_cr = emb42_cr + beta * torch.matmul(p_emb42_cr, proj_ma.T)
                emb43_cr = emb43_cr + beta * torch.matmul(p_emb43_cr, proj_ma.T)
                
            outputs[batch, pos11] = emb11_cr
            outputs[batch, pos12] = emb12_cr
            outputs[batch, pos13] = emb13_cr
            outputs[batch, pos21] = emb21_cr
            outputs[batch, pos22] = emb22_cr
            outputs[batch, pos23] = emb23_cr
            outputs[batch, pos31] = emb31_cr
            outputs[batch, pos32] = emb32_cr
            outputs[batch, pos33] = emb33_cr
            outputs[batch, pos41] = emb41_cr
            outputs[batch, pos42] = emb42_cr
            outputs[batch, pos43] = emb43_cr
        except RuntimeError:
            pass
        
    output = rearrange(
        outputs,
        "batch seq_len d_resid -> batch seq_len d_resid",
        d_resid=model.config.hidden_size,
    )
    torch.cuda.empty_cache()
    return (output,)


def eval_model_performance(model, dataloader, input_tp="table"):
    total_count = 0
    correct_count = 0
    total_logit = []
    model.eval()
    apply_softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        for _, output in tqdm(enumerate(tqdm(dataloader))):
            for k, v in output.items():
                if v is not None and isinstance(v, torch.Tensor):
                    output[k] = v.to(model.device)

            outputs = model(input_ids=output[f"base_tokens_{input_tp}"])

            for bi in range(output["labels"].size(0)):
                label = output["labels"][bi]
                pred = torch.argmax(
                    outputs.logits[bi][-1]
                )
                logit = apply_softmax(outputs.logits[bi][-1])[label].item()
                total_logit.append(logit)
                if label == pred:
                    correct_count += 1
                total_count += 1
    del outputs
    torch.cuda.empty_cache()

    current_acc = round(correct_count / total_count, 2)
    m_logit = sum(total_logit) / len(total_logit)
    return current_acc, m_logit


def act_patching_main_resid_ap(
        model: AutoModelForCausalLM=None,
        tokenizer: AutoTokenizer=None, 
        data_file: str="",
        num_sample: int=30,
        batch_size: int=30,
        l1: int=10,
        l2: int=20,
        vari: str="p_cr",
        input_tp: str="table",
        data_tp: str="space",
        enti: int=1,
        atti: int=2,
        proj_ma: torch.tensor=None,
        use_rand_proj_ma: int=0,
        patch_tp: int="modify",
        beta: float=0.55,
        nb_loop: int=1,):
    print(beta)
    accs = defaultdict(list)
    apply_softmax = torch.nn.Softmax(dim=-1)
    base_accs = []
    acc_cri = 0.25
    for loop_idx in tqdm(range(nb_loop)):
        acc = 0.0
        while acc < acc_cri:
            dataloader = load_dataloader(
                tokenizer=tokenizer,
                data_file=data_file,
                num_samples=num_sample,
                batch_size=batch_size,
                vari=vari,
                data_tp=data_tp,
                input_tp=input_tp,
                enti=enti,
                atti=atti,
            )

            acc, logit = eval_model_performance(model, dataloader, input_tp=input_tp)
        print(f"Model Acc.: {acc} Logit: {logit}")
    
        base_accs.append([round(acc * 100, 2), round(logit, 2)])
        
        correct_count, total_count = 0, 0
        total_logit = []
        for bi, inputs in enumerate(dataloader):
            # Step 1: Compute clean and corrupt caches
            (
                clean_cache,
                corrupt_cache,
                hook_points,
            ) = get_caches_resid_ap(model, dataloader, l1, l2, input_tp=input_tp)
        
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)

            source_cache = corrupt_cache
            # Step 2: Activation patching
            
            with TraceDict(
                    model,
                    hook_points,
                    retain_input=True,
                    edit_output=partial(
                        activation_patching_resid_ap,
                        model=model,
                        source_cache=source_cache,
                        bi=bi,
                        input_tokens=inputs,
                        input_tp=input_tp,
                        proj_ma=proj_ma,
                        use_rand_proj_ma=use_rand_proj_ma,
                        patch_tp=patch_tp,
                        beta=beta,
                    ),
            ) as _:
                outputs = model(inputs[f"base_tokens_{input_tp}"])
                    
            for idx in range(inputs[f"base_tokens_{input_tp}"].size(0)):
                label = inputs["labels"][idx].item()
                pred = torch.argmax(outputs.logits[idx, -1], dim=-1).item()

                logit = apply_softmax(outputs.logits[idx, -1])[label].item()
                total_logit.append(logit)
                
                if label == pred:
                    correct_count += 1
                total_count += 1

            del outputs
            torch.cuda.empty_cache()
            
        acc = round(correct_count / total_count * 100, 2)
        m_logit = sum(total_logit) / len(total_logit)
        accs[vari].append((acc, round(m_logit, 2)))
        

    result = {}
    for input_tp, lst_acc in accs.items():
        lst_acc = np.array(lst_acc)
        acc_m = np.mean(lst_acc[:,0])
        acc_std = np.std(lst_acc[:,0])
        logit_m = np.mean(lst_acc[:,1])
        logit_std = np.std(lst_acc[:,1])
        
        lst_acc = np.array(base_accs)
        acc_m_b = np.mean(lst_acc[:,0])
        acc_std_b = np.std(lst_acc[:,0])
        logit_m_b = np.mean(lst_acc[:,1])
        logit_std_b = np.std(lst_acc[:,1])
        rt = round(acc_m/acc_m_b, 2)
        print(f"Base: Accuracy: {acc_m_b} ({acc_std_b}) , Logit: {logit_m_b} ({logit_std_b})")
        print(f"Base + {vari}: Accuracy: {acc_m} ({acc_std}) , Logit: {logit_m} ({logit_std})")
        print(f"Rate: {rt}.")

        result["a"] = acc_m_b
        result["a_cr"] = acc_m
        result["astd"] = acc_std_b
        result["astd_cr"] = acc_std

        result["l"] = logit_m_b
        result["l_cr"] = logit_m
        result["lstd"] = logit_std_b
        result["lstd_cr"] = logit_std
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for activation patching")
    
    parser.add_argument("--llm_tp", type=str, default="llama", help="llama / qwen")
    parser.add_argument("--input_tp", type=str, default="story", help="table / temp / story")
    parser.add_argument("--vari_tp", type=str, default="p_cr", help="p_cr / r_cr")
    parser.add_argument("--space_tp", type=str, default="i", help="i / r")
    parser.add_argument("--cr_tp", type=str, default="acr", help="acr / ecr")
    parser.add_argument("--enti", type=int, default=1, help="0 / 1 / 2")
    parser.add_argument("--atti", type=int, default=1, help="1 / 2 / 3 / 4")
    parser.add_argument("--nb_loop", type=int, default=1, help="1 / 2 / 3")
    parser.add_argument("--use_rand_proj_ma", type=int, default=0, help="0 / 1")
    parser.add_argument("--patch_tp", type=str, default="modify", help="replace / modify")
    
    parser.add_argument("--l1", type=int, default=10, help="0 ~ 31")
    parser.add_argument("--l2", type=int, default=20, help="0 ~ 31")

    parser.add_argument("--num_sample", type=int, default=100, help="the number of test samples")
    parser.add_argument("--batch_size", type=int, default=50, help="the numebr of batch size")
    
    parser.add_argument("--steer_tp", type=str, default="steer_a_1_2", help="e.g., steer_a_1_2, steer_a_1_3, ..., steer_e_2_3,")
    args = parser.parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(args.llm_tp, device)
    print("Model and Tokenizer loaded")

    data_tps = ["space", "create", "job", "relation", "city"]

    sd_result = "./data/data_table_result_fewshot/"
    all_result = {}
    for data_tp in data_tps:
        sd_proj = "./data/data_table_emb/"
        sf_proj = sd_proj + f"{args.llm_tp}_l15_{args.space_tp}_{data_tp}_proj.npy"
        proj_ma = np.load(sf_proj)
        proj_ma = torch.tensor(proj_ma).to(device).to(torch.bfloat16)[:,:]
        
        sd = "./data/data_table/"
        sf_data = sd + f"{data_tp}_tts_all.jsonl"

        all_result[data_tp] = {}
        for beta in np.arange(0.1, 1.25, 0.05)[:]:
            #for beta in [-2.5, -2.6, -2.7, -2.8, -2.9, -3.0]:
            print(f"Patching {data_tp} format {args.input_tp} via {args.vari_tp} beta={beta}...")
            result = act_patching_main_resid_ap(model=model,
                                                tokenizer=tokenizer,
                                                data_file=sf_data,
                                                num_sample=args.num_sample,
                                                batch_size=args.batch_size,
                                                l1 = args.l1,
                                                l2 = args.l2,
                                                vari=args.vari_tp,
                                                input_tp=args.input_tp,
                                                data_tp=data_tp,
                                                enti=args.enti,
                                                atti=args.atti,
                                                proj_ma=proj_ma,
                                                use_rand_proj_ma=args.use_rand_proj_ma,
                                                patch_tp=args.patch_tp,
                                                beta=beta,
                                                nb_loop=args.nb_loop,)

            all_result[data_tp][beta] = result

    if args.use_rand_proj_ma == 1:
        ma_tp = "randma"
    else:
        ma_tp = "projma"
    with open(sd_result + f"ap_result_{args.llm_tp}_{args.input_tp}_atti_{args.atti}_{ma_tp}.json", 'w') as fout:
        json.dump(all_result, fout)
        
