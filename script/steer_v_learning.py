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
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_name == "qwen":
        model_id = "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,)
    return model, tokenizer


def load_tb_data(tokenizer, num_samples, data_file, rand=True,
                 data_tp="space", input_tp="story"):
    
    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
            
    if rand:
        random.shuffle(data)

    que_temp = "Based on the context, given like {ENT1} to {ATT1} , {ENT2} to"

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
    
    prompts_temp = []
    prompts_table = []
    prompts_story = []
    # where a2 means the atti=2 in one shot part.
    prompts_temp_a2 = []
    prompts_table_a2 = []
    prompts_story_a2 = []

    prompts_temp_a3 = []
    prompts_table_a3 = []
    prompts_story_a3 = []

    prompts_temp_a4 = []
    prompts_table_a4 = []
    prompts_story_a4 = []

    # where e2 means the enti=2 in last part e.g., "pot to?".
    prompts_temp_e2 = []
    prompts_table_e2 = []
    prompts_story_e2 = []

    for i in range(num_samples):
        enti_shot = 0
        atti_shot = 1
        enti = 1
        
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

        ## shot part atti=1
        att_tp_a1 = f"atts{atti_shot}"
        ent1_b = data[i]["ents"][enti_shot]
        att1_b = data[i][att_tp_a1][enti_shot]
        ent2_b = data[i]["ents"][enti]
        att2 = data[i][att_tp_a1][enti]
        que = que_temp.replace("{ENT1}", ent1_b).replace("{ATT1}", att1_b).replace("{ENT2}", ent2_b)
        labels.append(tokenizer.encode(" %s"%att2)[-1])
        prompts_table.append(ctx_t + que)
        prompts_temp.append(ctx + que)
        prompts_story.append(ctx_s + que)
        ##
        ## shot part atti=2 
        att_tp_a2 = f"atts{atti_shot+1}"
        ent1 = data[i]["ents"][enti_shot]
        att1 = data[i][att_tp_a2][enti_shot]
        ent2 = data[i]["ents"][enti]
        att2 = data[i][att_tp_a2][enti]
        que_a2 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1).replace("{ENT2}", ent2)

        ctx_t_a2 = ctx_t.replace(att1_b, "PLACE")
        ctx_t_a2 = ctx_t_a2.replace(att1, att1_b)
        ctx_t_a2 = ctx_t_a2.replace("PLACE", att1)
        
        ctx_a2 = ctx.replace(att1_b, "PLACE")
        ctx_a2 = ctx_a2.replace(att1, att1_b)
        ctx_a2 = ctx_a2.replace("PLACE", att1)

        ctx_s_a2 = ctx_s.replace(att1_b, "PLACE")
        ctx_s_a2 = ctx_s_a2.replace(att1, att1_b)
        ctx_s_a2 = ctx_s_a2.replace("PLACE", att1)

        que_a2 = que_a2.replace(att1, att1_b)
        prompts_table_a2.append(ctx_t_a2 + que_a2)
        prompts_temp_a2.append(ctx_a2 + que_a2)
        prompts_story_a2.append(ctx_s_a2 + que_a2)
        ##
        ## shot part atti=3 
        att_tp_a3 = f"atts{atti_shot+2}"
        ent1 = data[i]["ents"][enti_shot]
        att1 = data[i][att_tp_a3][enti_shot]
        ent2 = data[i]["ents"][enti]
        att2 = data[i][att_tp_a3][enti]
        que_a3 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1).replace("{ENT2}", ent2)

        ctx_t_a3 = ctx_t.replace(att1_b, "PLACE")
        ctx_t_a3 = ctx_t_a3.replace(att1, att1_b)
        ctx_t_a3 = ctx_t_a3.replace("PLACE", att1)
        
        ctx_a3 = ctx.replace(att1_b, "PLACE")
        ctx_a3 = ctx_a3.replace(att1, att1_b)
        ctx_a3 = ctx_a3.replace("PLACE", att1)

        ctx_s_a3 = ctx_s.replace(att1_b, "PLACE")
        ctx_s_a3 = ctx_s_a3.replace(att1, att1_b)
        ctx_s_a3 = ctx_s_a3.replace("PLACE", att1)

        que_a3 = que_a3.replace(att1, att1_b)
        prompts_table_a3.append(ctx_t_a3 + que_a3)
        prompts_temp_a3.append(ctx_a3 + que_a3)
        prompts_story_a3.append(ctx_s_a3 + que_a3)
        
        ##
        ## shot part atti=4
        att_tp_a4 = f"atts{atti_shot+3}"
        ent1 = data[i]["ents"][enti_shot]
        att1 = data[i][att_tp_a4][enti_shot]
        ent2 = data[i]["ents"][enti]
        att2 = data[i][att_tp_a4][enti]
        que_a4 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1).replace("{ENT2}", ent2)

        ctx_t_a4 = ctx_t.replace(att1_b, "PLACE")
        ctx_t_a4 = ctx_t_a4.replace(att1, att1_b)
        ctx_t_a4 = ctx_t_a4.replace("PLACE", att1)
        
        ctx_a4 = ctx.replace(att1_b, "PLACE")
        ctx_a4 = ctx_a4.replace(att1, att1_b)
        ctx_a4 = ctx_a4.replace("PLACE", att1)

        ctx_s_a4 = ctx_s.replace(att1_b, "PLACE")
        ctx_s_a4 = ctx_s_a4.replace(att1, att1_b)
        ctx_s_a4 = ctx_s_a4.replace("PLACE", att1)

        que_a4 = que_a4.replace(att1, att1_b)
        prompts_table_a4.append(ctx_t_a4 + que_a4)
        prompts_temp_a4.append(ctx_a4 + que_a4)
        prompts_story_a4.append(ctx_s_a4 + que_a4)
        ##
        ## last part enti=2
        att_tp = f"atts{atti_shot}"
        enti_e2 = enti + 1
        ent1 = data[i]["ents"][enti_shot]
        att1 = data[i][att_tp][enti_shot]
        ent2 = data[i]["ents"][enti_e2]
        que_e2 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1).replace("{ENT2}", ent2)

        ctx_t_e2 = ctx_t.replace(ent2_b, "PLACE")
        ctx_t_e2 = ctx_t_e2.replace(ent2, ent2_b)
        ctx_t_e2 = ctx_t_e2.replace("PLACE", ent2)
        
        ctx_e2 = ctx.replace(ent2_b, "PLACE")
        ctx_e2 = ctx_e2.replace(ent2, ent2_b)
        ctx_e2 = ctx_e2.replace("PLACE", ent2)
      	
        ctx_s_e2 = ctx_s.replace(ent2_b, "PLACE")
        ctx_s_e2 = ctx_s_e2.replace(ent2, ent2_b)
        ctx_s_e2 = ctx_s_e2.replace("PLACE", ent2)

        que_e2 = que_e2.replace(ent2, ent2_b)
        prompts_table_e2.append(ctx_t_e2 + que_e2)
        prompts_temp_e2.append(ctx_e2 + que_e2)
        prompts_story_e2.append(ctx_s_e2 + que_e2)
        ##
        
    ## shot part atti = 1
    input_tokens_table = tokenizer(prompts_table, padding=True, return_tensors="pt")
    input_ids_table = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp, padding=True, return_tensors="pt")
    input_ids_temp = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story, padding=True, return_tensors="pt")
    input_ids_story = input_tokens_story["input_ids"]
    ## shot part atti = 2
    input_tokens_table = tokenizer(prompts_table_a2, padding=True, return_tensors="pt")
    input_ids_table_a2 = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp_a2, padding=True, return_tensors="pt")
    input_ids_temp_a2 = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story_a2, padding=True, return_tensors="pt")
    input_ids_story_a2 = input_tokens_story["input_ids"]
    ## shot part atti = 3
    input_tokens_table = tokenizer(prompts_table_a3, padding=True, return_tensors="pt")
    input_ids_table_a3 = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp_a3, padding=True, return_tensors="pt")
    input_ids_temp_a3 = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story_a3, padding=True, return_tensors="pt")
    input_ids_story_a3 = input_tokens_story["input_ids"]
    ## shot part atti = 4
    input_tokens_table = tokenizer(prompts_table_a4, padding=True, return_tensors="pt")
    input_ids_table_a4 = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp_a4, padding=True, return_tensors="pt")
    input_ids_temp_a4 = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story_a4, padding=True, return_tensors="pt")
    input_ids_story_a4 = input_tokens_story["input_ids"]
    ## last part enti = 2  
    input_tokens_table = tokenizer(prompts_table_e2, padding=True, return_tensors="pt")
    input_ids_table_e2 = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp_e2, padding=True, return_tensors="pt")
    input_ids_temp_e2 = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story_e2, padding=True, return_tensors="pt")
    input_ids_story_e2 = input_tokens_story["input_ids"]
    
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

    return (input_ids_table, input_ids_temp, input_ids_story,
            ents1, ents2, ents3,
            atts1_1, atts1_2, atts1_3,
            atts2_1, atts2_2, atts2_3,
            atts3_1, atts3_2, atts3_3,
            atts4_1, atts4_2, atts4_3,
            labels,
            input_ids_table_a2, input_ids_temp_a2, input_ids_story_a2,
            input_ids_table_a3, input_ids_temp_a3, input_ids_story_a3,
            input_ids_table_a4, input_ids_temp_a4, input_ids_story_a4,
            input_ids_table_e2, input_ids_temp_e2, input_ids_story_e2,)


def load_dataloader(
        tokenizer: AutoTokenizer,
        lst_data_file: list,
        num_samples: int,
        batch_size: int,
        data_tp: str="space",
        input_tp: str="story",):
    
    raw_data = load_tb_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        lst_data_file=lst_data_file,
        data_tp=data_tp,
        input_tp=input_tp,
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

    labels = raw_data[18]

    base_tokens_table_a2 = raw_data[19]
    base_tokens_temp_a2 = raw_data[20]
    base_tokens_story_a2 = raw_data[21]

    base_tokens_table_a3 = raw_data[22]
    base_tokens_temp_a3 = raw_data[23]
    base_tokens_story_a3 = raw_data[24]

    base_tokens_table_a4 = raw_data[25]
    base_tokens_temp_a4 = raw_data[26]
    base_tokens_story_a4 = raw_data[27]

    base_tokens_table_e2 = raw_data[28]
    base_tokens_temp_e2 = raw_data[29]
    base_tokens_story_e2 = raw_data[30]
    
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
            "labels": labels,
            "base_tokens_table_a2": base_tokens_table_a2,
            "base_tokens_temp_a2": base_tokens_temp_a2,
            "base_tokens_story_a2": base_tokens_story_a2,
            "base_tokens_table_a3": base_tokens_table_a3,
            "base_tokens_temp_a3": base_tokens_temp_a3,
            "base_tokens_story_a3": base_tokens_story_a3,
            "base_tokens_table_a4": base_tokens_table_a4,
            "base_tokens_temp_a4": base_tokens_temp_a4,
            "base_tokens_story_a4": base_tokens_story_a4,
            "base_tokens_table_e2": base_tokens_table_e2,
            "base_tokens_temp_e2": base_tokens_temp_e2,
            "base_tokens_story_e2": base_tokens_story_e2,
        }
    ).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_x_y(
        model: AutoModelForCausalLM,
        dataloader: torch.utils.data.DataLoader,
        layer: int=15,
        input_tp: str="table",
        end_pos_a: int=-4,
        end_pos_e: int=-2,):
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

    labels = []

    xs_a1 = []
    xs_a2 = []
    xs_a3 = []
    xs_a4 = []

    xs_e2 = []#where enti=1, enti=0 is for shot.
    xs_e3 = []#where enti=2
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

                xs_a1.append(cache[f"model.layers.{layer}"].output[i][[end_pos_a]])
                xs_e2.append(cache[f"model.layers.{layer}"].output[i][[end_pos_e]])
                
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

                labels.append(inp["labels"][i])
                
            del cache
            torch.cuda.empty_cache()

            for ai in ["a2", "a3", "a4"]:
                with TraceDict(
                        model,
                        hook_points,
                        retain_input=True,
                ) as cache:
                    output = model(inp[f"base_tokens_{input_tp}_{ai}"])

                for k, v in cache.items():
                    if v is not None and isinstance(v, nethook.Trace):
                        cache[k].input = v.input.to("cpu")
                        cache[k].output = v.output[0].to("cpu")

                for i in range(batch_size):
                    if ai == "a2":
                        xs_a2.append(cache[f"model.layers.{layer}"].output[i][[end_pos_a]])
                    elif ai == "a3":
                        xs_a3.append(cache[f"model.layers.{layer}"].output[i][[end_pos_a]])
                    elif ai == "a4":
                        xs_a4.append(cache[f"model.layers.{layer}"].output[i][[end_pos_a]])
                del cache
                torch.cuda.empty_cache()
                
            for ei in ["e2",]:
                with TraceDict(
                        model,
                        hook_points,
                        retain_input=True,
                ) as cache:
                    output = model(inp[f"base_tokens_{input_tp}_{ei}"])

                for k, v in cache.items():
                    if v is not None and isinstance(v, nethook.Trace):
                        cache[k].input = v.input.to("cpu")
                        cache[k].output = v.output[0].to("cpu")

                for i in range(batch_size):
                    xs_e3.append(cache[f"model.layers.{layer}"].output[i][[end_pos_e]])
                del cache
                torch.cuda.empty_cache()

            del inp
            
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

    labels = torch.stack(labels)
    
    xs_a1 = torch.cat(xs_a1, 0).float()
    xs_a2 = torch.cat(xs_a2, 0).float()
    xs_a3 = torch.cat(xs_a3, 0).float()
    xs_a4 = torch.cat(xs_a4, 0).float()

    xs_e2 = torch.cat(xs_e2, 0).float()
    xs_e3 = torch.cat(xs_e3, 0).float()
    
    assert xs_a1.size(0) == xs_a2.size(0) == xs_a3.size(0) == xs_a4.size(0) 
    assert xs_e2.size(0) == xs_e3.size(0)
    
    return (xs_ents1, xs_ents2, xs_ents3,
            xs_atts1_1, xs_atts1_2, xs_atts1_3,
            xs_atts2_1, xs_atts2_2, xs_atts2_3,
            xs_atts3_1, xs_atts3_2, xs_atts3_3,
            xs_atts4_1, xs_atts4_2, xs_atts4_3,
            labels,
            xs_a1, xs_a2, xs_a3, xs_a4,
            xs_e2, xs_e3,)

def pls_projection(lst_xs=[], base=0, n_components=1):
    # Combine data
    
    X = torch.cat(lst_xs, dim=0)  # (n1+n2, d)
    Y = torch.cat([
        (i+1+base)*torch.ones(xs.size(0), 1) for i, xs in enumerate(lst_xs)
    ], dim=0)  # (n1+n2, 1)
    
    # Center X and Y
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Cross-covariance matrix
    C = Xc.T @ Yc  # (d, 1)

    # For 1 component: weight vector w is the normalized covariance direction
    w = C / torch.norm(C)

    # For multi-component version (simplified deflation loop)
    if n_components > 1:
        Ws = []
        Xr = Xc.clone()
        Yr = Yc.clone()
        for _ in range(n_components):
            C = Xr.T @ Yr
            w = C / torch.norm(C)
            t = Xr @ w
            p = (Xr.T @ t) / (t.T @ t)
            q = (Yr.T @ t) / (t.T @ t)
            Xr = Xr - t @ p.T
            Yr = Yr - t @ q
            Ws.append(w)
        W = torch.cat(Ws, dim=1)
    else:
        W = w  # (d, 1)
    return W, X_mean, Y_mean


def extract_steer_vector(
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        datafile: str,
        sfout: str,
        num_samples: int,
        layer: int=15,
        input_tp: str="table",
        batch_size: int=100,
        proj_ma: torch.tensor=None,
        pls_dim: int=30,
        ):

    dataloader_tb = load_dataloader(tokenizer, datafile, num_samples, batch_size=batch_size)
    (xs_ents1, xs_ents2, xs_ents3,
     xs_atts1_1, xs_atts1_2, xs_atts1_3,
     xs_atts2_1, xs_atts2_2, xs_atts2_3,
     xs_atts3_1, xs_atts3_2, xs_atts3_3,
     xs_atts4_1, xs_atts4_2, xs_atts4_3,
     labels,
     xs_a1, xs_a2, xs_a3, xs_a4,
     xs_e2, xs_e3,) = get_x_y(model, dataloader_tb, layer, input_tp,)

    proj_ma, _, _ = pls_projection([xs_a1, xs_a2, xs_a3, xs_a4], n_components=pls_dim)
    
    xs_a1 = xs_a1 @ proj_ma
    xs_a2 = xs_a2 @ proj_ma
    xs_a3 = xs_a3 @ proj_ma
    xs_a4 = xs_a4 @ proj_ma

    proj_ma_e, _, _ = pls_projection([xs_e2, xs_e3], base=1, n_components=pls_dim)
    xs_e2 = xs_e2 @ proj_ma_e
    xs_e3 = xs_e3 @ proj_ma_e
    
    steer_a_1_2 = (xs_a2 - xs_a1).mean(dim=0)
    steer_a_1_3 = (xs_a3 - xs_a1).mean(dim=0)
    steer_a_1_4 = (xs_a4 - xs_a1).mean(dim=0)
    steer_a_2_3 = (xs_a3 - xs_a2).mean(dim=0)
    steer_a_2_4 = (xs_a4 - xs_a2).mean(dim=0)
    steer_a_3_4 = (xs_a4 - xs_a3).mean(dim=0)
    
    steer_e_2_3 = (xs_e3 - xs_e2).mean(dim=0)
    
    data_out = {
        "steer_a_1_2": steer_a_1_2 @ proj_ma.T,
        "steer_a_1_3": steer_a_1_3 @ proj_ma.T,
        "steer_a_1_4": steer_a_1_4 @ proj_ma.T,
        "steer_a_2_3": steer_a_2_3 @ proj_ma.T,
        "steer_a_2_4": steer_a_2_4 @ proj_ma.T,
        "steer_a_3_4": steer_a_3_4 @ proj_ma.T,
        "steer_e_2_3": steer_e_2_3 @ proj_ma_e.T,
        }
    torch.save(data_out, sfout)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for activation patching")
    parser.add_argument("--llm_tp", type=str, default="llama", help="llama / qwen")
    parser.add_argument("--pls_dim", type=int, default=5, help="3, 4, 5, ...")
    parser.add_argument("--num_samples", type=int, default=500, help="the number of samples")
    parser.add_argument("--batch_size", type=int, default=250, help="the batch size")
    args = parser.parse_args()
    
    llm_tp = args.llm_tp
    pls_dim = args.pls_dim
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(llm_tp, device)
    print("Model and Tokenizer loaded")
    sd_in = "./data/"
    sd_out = "./data_emb/"

    num_samples = args.num_samples
    batch_size = args.batch_size
    
    for data_tp in ["city", "create", "job", "relation", "space"][:]:
        for input_tp in ["story", "temp", "table"][1:]:
            print(f"Processing {data_tp} dataset {input_tp} input ...")
            datafile = sd_in + f"{data_tp}_tts_all.jsonl"
            
            layers1 = [12, 13, 14, 15, 16,]
            layers2 = [11, 17, 18, 19, 20]
            layers = layers1 + layers2
            for layer in layers:
                sfout = sd_out + f"{llm_tp}_l{layer}_{data_tp}_{input_tp}_steer.pt"
                extract_steer_vector(
                    tokenizer,
                    model,
                    datafile,
                    sfout,
                    num_samples,
                    layer=int(layer),
                    input_tp=input_tp,
                    batch_size=batch_size,
                    pls_dim=pls_dim,
                )

    
