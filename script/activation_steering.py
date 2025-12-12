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
                 data_tp="space", input_tp="story", enti=1, atti=2, step=1, cr_tp="acr"):
    
    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if rand:
        random.shuffle(data)

    que_temp = "Based on the context, given like {ENT1} to {ATT1} , {ENT2} to"
    att_tp = f"atts{atti}"
    if (atti + step) <= 4:
        att_tp_acr = f"atts{atti+step}"
    else:
        att_tp_acr = f"atts{atti+step-4}"

    if enti == 0:
        enti_shot = 2
        enti_ecr = 1
    elif enti == 1:
        enti_shot = 0
        enti_ecr = 2
    elif enti == 2:
        enti_shot = 1
        enti_ecr = 0
        
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
    labels_acr = []
    labels_ecr = []

    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    
    prompts_temp = []
    prompts_table = []
    prompts_story = []

    prompts_temp_acr = []
    prompts_table_acr = []
    prompts_story_acr = []

    prompts_temp_ecr = []
    prompts_table_ecr = []
    prompts_story_ecr = []
    
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

        ent1_b = data[i]["ents"][enti_shot]
        att1_b = data[i][att_tp][enti_shot]
        ent2_b = data[i]["ents"][enti]
        att2 = data[i][att_tp][enti]
        que = que_temp.replace("{ENT1}", ent1_b).replace("{ATT1}", att1_b).replace("{ENT2}", ent2_b)
        labels.append(tokenizer.encode(" %s"%att2)[-1])
        prompts_table.append(ctx_t + que)

        prompts_temp.append(ctx + que)
        prompts_story.append(ctx_s + que)
        
        ent1 = data[i]["ents"][enti_shot]
        att1 = data[i][att_tp_acr][enti_shot]
        ent2 = data[i]["ents"][enti]
        att2 = data[i][att_tp_acr][enti]
        que_acr = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1).replace("{ENT2}", ent2)
        labels_acr.append(tokenizer.encode(" %s"%att2)[-1])

        ctx_acr = ctx.replace(att1_b, "PLACE")
        ctx_acr = ctx_acr.replace(att1, att1_b)
        ctx_acr = ctx_acr.replace("PLACE", att1)

        ctx_s_acr = ctx_s.replace(att1_b, "PLACE")
        ctx_s_acr = ctx_s_acr.replace(att1, att1_b)
        ctx_s_acr = ctx_s_acr.replace("PLACE", att1)

        que_acr = que_acr.replace(att1, att1_b)
        prompts_table_acr.append(ctx_t + que_acr)
        prompts_temp_acr.append(ctx_acr + que_acr)
        prompts_story_acr.append(ctx_s_acr + que_acr)

        ent1 = data[i]["ents"][enti_shot]
        att1 = data[i][att_tp][enti_shot]
        ent2 = data[i]["ents"][enti_ecr]
        att2 = data[i][att_tp][enti_ecr]
        que_ecr = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1).replace("{ENT2}", ent2)
        labels_ecr.append(tokenizer.encode(" %s"%att2)[-1])

        ctx_ecr = ctx.replace(ent2_b, "PLACE")
        ctx_ecr = ctx_ecr.replace(ent2, ent2_b)
        ctx_ecr = ctx_ecr.replace("PLACE", ent2)
      	
        ctx_s_ecr = ctx_s.replace(ent2_b, "PLACE")
        ctx_s_ecr = ctx_s_ecr.replace(ent2, ent2_b)
        ctx_s_ecr = ctx_s_ecr.replace("PLACE", ent2)

        que_ecr = que_acr.replace(ent2, ent2_b)
        prompts_table_ecr.append(ctx_t + que_ecr)
        prompts_temp_ecr.append(ctx_ecr + que_ecr)
        prompts_story_ecr.append(ctx_s_ecr + que_ecr)
        
        """
        labels1.append(tokenizer.encode(" %s"%data[i]["atts1"][enti])[-1])
        labels2.append(tokenizer.encode(" %s"%data[i]["atts2"][enti])[-1])
        labels3.append(tokenizer.encode(" %s"%data[i]["atts3"][enti])[-1])
        labels4.append(tokenizer.encode(" %s"%data[i]["atts4"][enti])[-1])
        """
    input_tokens_table = tokenizer(prompts_table, padding=True, return_tensors="pt")
    input_ids_table = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp, padding=True, return_tensors="pt")
    input_ids_temp = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story, padding=True, return_tensors="pt")
    input_ids_story = input_tokens_story["input_ids"]

    input_tokens_table = tokenizer(prompts_table_ecr, padding=True, return_tensors="pt")
    input_ids_table_ecr = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp_ecr, padding=True, return_tensors="pt")
    input_ids_temp_ecr = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story_ecr, padding=True, return_tensors="pt")
    input_ids_story_ecr = input_tokens_story["input_ids"]

    input_tokens_table = tokenizer(prompts_table_acr, padding=True, return_tensors="pt")
    input_ids_table_acr = input_tokens_table["input_ids"]
    input_tokens_temp = tokenizer(prompts_temp_acr, padding=True, return_tensors="pt")
    input_ids_temp_acr = input_tokens_temp["input_ids"]
    input_tokens_story = tokenizer(prompts_story_acr, padding=True, return_tensors="pt")
    input_ids_story_acr = input_tokens_story["input_ids"]
    
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
    labels_acr = torch.tensor(labels_acr)
    labels_ecr = torch.tensor(labels_ecr)
    
    return (input_ids_table, input_ids_temp, input_ids_story,
            ents1, ents2, ents3,
            atts1_1, atts1_2, atts1_3,
            atts2_1, atts2_2, atts2_3,
            atts3_1, atts3_2, atts3_3,
            atts4_1, atts4_2, atts4_3,
            labels, labels_acr, labels_ecr,
            input_ids_table_acr, input_ids_temp_acr, input_ids_story_acr,
            input_ids_table_ecr, input_ids_temp_ecr, input_ids_story_ecr,)


def load_dataloader(
        tokenizer: AutoTokenizer,
        data_file: list,
        num_samples: int,
        batch_size: int,
        data_tp: str="space",
        input_tp: str="story",
        enti: int=1,
        atti: int=1,
        step: int=1,):
    
    raw_data = load_tb_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=data_file,
        data_tp=data_tp,
        input_tp=input_tp,
        enti=enti,
        atti=atti,
        step=step,
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
    labels_acr = raw_data[19]
    labels_ecr = raw_data[20]

    base_tokens_table_acr = raw_data[21]
    base_tokens_temp_acr = raw_data[22]
    base_tokens_story_acr = raw_data[23]

    base_tokens_table_ecr = raw_data[24]
    base_tokens_temp_ecr = raw_data[25]
    base_tokens_story_ecr = raw_data[26]

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
            "labels_acr": labels_acr,
            "labels_ecr": labels_ecr,
            "base_tokens_table_acr": base_tokens_table_acr,
            "base_tokens_temp_acr": base_tokens_temp_acr,
            "base_tokens_story_acr": base_tokens_story_acr,
            "base_tokens_table_ecr": base_tokens_table_ecr,
            "base_tokens_temp_ecr": base_tokens_temp_ecr,
            "base_tokens_story_ecr": base_tokens_story_ecr,
            
        }
    ).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_caches_resid_ap(
        model: AutoModelForCausalLM,
        dataloader: torch.utils.data.DataLoader,
        layer1=10,
        layer2=20,
        input_tp="table",
        cr_tp="acr",):
    
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
                output = model(inp[f"base_tokens_{input_tp}_{cr_tp}"])

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
        beta: float=0.55,
        pos_que_ent: int=-2,
        pos_que_att: int=-4,
        step: int=1,
        cr_tp: str="acr",
        steer_vs: dict=None,
        steer_tp: str="steer_a_1_2",
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
        try:
            if cr_tp == "acr":
                pos_k = pos_que_att
            elif cr_tp == "ecr":
                pos_k = pos_que_ent
                
            emb = outputs[batch, pos_k]
            emb_cr = cache[batch, pos_k]
            steer_v = steer_vs[steer_tp].to(emb)
            emb_new = emb + beta * steer_v
            outputs[batch, pos_k] = emb_new
            
        except RuntimeError:
            pass
        
    output = rearrange(
        outputs,
        "batch seq_len d_resid -> batch seq_len d_resid",
        d_resid=model.config.hidden_size,
    )
    torch.cuda.empty_cache()
    return (output,)


def eval_model_performance(model, dataloader, input_tp="table", label_tp="labels"):
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

            for bi in range(output[label_tp].size(0)):
                label = output[label_tp][bi]
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
        cr_tp: int="acr",
        beta: float=0.55,
        step: int=1,
        steer_vs: dict=None,
        steer_tp: str="steer_a_1_2",
        nb_loop: int=1,):
    print(beta)
    accs = defaultdict(list)
    accs_ap = defaultdict(list)
    apply_softmax = torch.nn.Softmax(dim=-1)
    base_accs = []
    base_accs_cr = []
    acc_cri = 0.25
    for loop_idx in tqdm(range(nb_loop)):
        acc = 0.0
        while acc < acc_cri:
            dataloader = load_dataloader(
                tokenizer=tokenizer,
                data_file=data_file,
                num_samples=num_sample,
                batch_size=batch_size,
                data_tp=data_tp,
                input_tp=input_tp,
                enti=enti,
                atti=atti,
            )

            acc, logit = eval_model_performance(model, dataloader, input_tp=input_tp)
            acc_cr, logit_cr = eval_model_performance(model, dataloader, input_tp=input_tp, label_tp=f"labels_{cr_tp}")
        print(f"Model Acc.: {acc} Logit: {logit}")
        print(f"Model Acc. (cr): {acc_cr} Logit (cr): {logit_cr}")
        base_accs.append([round(acc * 100, 2), round(logit, 2)])
        base_accs_cr.append([round(acc_cr * 100, 2), round(logit_cr, 2)])
        
        correct_count, total_count = 0, 0
        total_logit = []

        correct_count_ap, total_count_ap = 0, 0
        total_logit_ap = []
        for bi, inputs in enumerate(dataloader):
            # Step 1: Compute clean and corrupt caches
            (
                clean_cache,
                corrupt_cache,
                hook_points,
            ) = get_caches_resid_ap(model, dataloader, l1, l2, input_tp=input_tp, cr_tp=cr_tp)
        
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
                        beta=beta,
                        step=step,
                        cr_tp=cr_tp,
                        steer_vs=steer_vs,
                        steer_tp=steer_tp,
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

                if cr_tp == "acr":
                    label_ap = inputs[f"labels_acr"][idx].item()
                else:
                    label_ap = inputs[f"labels_ecr"][idx].item()
                logit_ap = apply_softmax(outputs.logits[idx, -1])[label_ap].item()
                total_logit_ap.append(logit_ap)
                
                if label_ap == pred:
                    correct_count_ap += 1
            del outputs
            torch.cuda.empty_cache()
            
        acc = round(correct_count / total_count * 100, 2)
        m_logit = sum(total_logit) / len(total_logit)
        accs[vari].append((acc, round(m_logit, 2)))

        acc = round(correct_count_ap / total_count * 100, 2)
        m_logit = sum(total_logit_ap) / len(total_logit_ap)
        accs_ap[vari].append((acc, round(m_logit, 2)))

    result = {}
    for input_tp, lst_acc in accs.items():
        lst_acc = np.array(lst_acc)
        acc_m = np.mean(lst_acc[:,0])
        acc_std = np.std(lst_acc[:,0])
        logit_m = np.mean(lst_acc[:,1])
        logit_std = np.std(lst_acc[:,1])
        
        lst_acc = np.array(accs_ap[input_tp])
        acc_m_ap = np.mean(lst_acc[:,0])
        acc_std_ap = np.std(lst_acc[:,0])
        logit_m_ap = np.mean(lst_acc[:,1])
        logit_std_ap = np.std(lst_acc[:,1])

        lst_acc = np.array(base_accs)
        acc_m_b = np.mean(lst_acc[:,0])
        acc_std_b = np.std(lst_acc[:,0])
        logit_m_b = np.mean(lst_acc[:,1])
        logit_std_b = np.std(lst_acc[:,1])
        rt = round(acc_m/acc_m_b, 2)

        lst_acc = np.array(base_accs_cr)
        acc_m_b_cr = np.mean(lst_acc[:,0])
        acc_std_b_cr = np.std(lst_acc[:,0])
        logit_m_b_cr = np.mean(lst_acc[:,1])
        logit_std_b_cr = np.std(lst_acc[:,1])
        
        print(f"Base (original): Accuracy: {acc_m_b} ({acc_std_b}) , Logit: {logit_m_b} ({logit_std_b})")
        print(f"Base (target): Accuracy : {acc_m_b_cr} ({acc_std_b_cr}) , Logit: {logit_m_b_cr} ({logit_std_b_cr})")
        print(f"Base (orginal) + AP: Accuracy: {acc_m} ({acc_std}) , Logit: {logit_m} ({logit_std})")
        print(f"Base (target) + AP: Accuracy: {acc_m_ap} ({acc_std_ap}) , Logit: {logit_m_ap} ({logit_std_ap})")

        result["a"] = acc_m_b
        result["a_cr"] = acc_m_b_cr
        result["a_ap"] = acc_m
        result["a_cr_ap"] = acc_m_ap
        result["astd"] = acc_std_b
        result["astd_cr"] = acc_std_b_cr
        result["astd_ap"] = acc_std
        result["astd_cr_ap"] = acc_std_ap

        result["l"] = logit_m_b
        result["l_cr"] = logit_m_b_cr
        result["l_ap"] = logit_m
        result["l_cr_ap"] = logit_m_ap
        result["lstd"] = logit_std_b
        result["lstd_cr"] = logit_std_b_cr
        result["lstd_ap"] = logit_std
        result["lstd_cr_ap"] = logit_std_ap

    return result


def mean_steer_vs(lst_steer_vs):
    all_steer_vs = defaultdict(list)
    for steer_vs in lst_steer_vs:
        for k, v in steer_vs.items():
            all_steer_vs[k].append(v)

    steer_v_out = {}
    for k, vs in all_steer_vs.items():
        v = torch.stack(vs).mean(dim=0)
        steer_v_out[k] = v
    return steer_v_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for activation patching")
    
    parser.add_argument("--llm_tp", type=str, default="llama", help="llama / qwen")
    parser.add_argument("--input_tp", type=str, default="story", help="table / temp / story")
    parser.add_argument("--cr_tp", type=str, default="acr", help="acr / ecr")
    parser.add_argument("--enti", type=int, default=1, help="0 / 1 / 2")
    parser.add_argument("--atti", type=int, default=1, help="1 / 2 / 3 / 4")
    parser.add_argument("--step", type=int, default=1, help="1 / 2 / 3")
    parser.add_argument("--use_rand_proj_ma", type=int, default=0, help="0 / 1")

    parser.add_argument("--l1", type=int, default=10, help="0 ~ 31")
    parser.add_argument("--l2", type=int, default=20, help="0 ~ 31")
    parser.add_argument("--num_sample", type=int, default=150, help="")
    parser.add_argument("--batch_size", type=int, default=75, help="")
    parser.add_argument("--nb_loop", type=int, default=1, help="")
    
    parser.add_argument("--steer_tp", type=str, default="steer_a_1_2", help="e.g., steer_a_1_2, steer_a_1_3, ..., steer_e_2_3,")
    parser.add_argument("--mean_steer_tp", type=str, default="context", help="all, context")
    args = parser.parse_args()
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(args.llm_tp, device)
    print("Model and Tokenizer loaded")

    data_tps = ["space", "create", "job", "relation", "city"]
    sd_proj = "./data_emb/"
    sd_result = "./result/steer/"
    all_result = {}
    lis = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for data_tp in data_tps:
        lst_steer_vs = []
        if args.mean_steer_tp == "all":
            for data_tp_ in data_tps:
                for li in lis:
                    sf_steer = sd_proj + f"{args.llm_tp}_l{li}_{data_tp_}_{args.input_tp}_steer.pt"
                    steer_vs = torch.load(sf_steer)
                    lst_steer_vs.append(steer_vs)
        elif args.mean_steer_tp == "context":
            for li in lis:
                sf_steer = sd_proj + f"{args.llm_tp}_l{li}_{data_tp}_{args.input_tp}_steer.pt"
                steer_vs = torch.load(sf_steer)
                lst_steer_vs.append(steer_vs)
        steer_vs = mean_steer_vs(lst_steer_vs)
            
        sd = "./data/"
        sf_data = sd + f"{data_tp}_tts_all.jsonl"

        max_tar_acc = 0.0
        for beta in np.arange(0.15, 1.25, 0.15)[:]:
            print(f"Patching {data_tp} format {args.input_tp} via {args.cr_tp} beta={beta}...")
            result = act_patching_main_resid_ap(model=model,
                                                tokenizer=tokenizer,
                                                data_file=sf_data,
                                                input_tp=args.input_tp,
                                                data_tp=data_tp,
                                                nb_loop=args.nb_loop,
                                                num_sample=args.num_sample,
                                                batch_size=args.batch_size,
                                                enti=args.enti,
                                                atti=args.atti,
                                                use_rand_proj_ma=args.use_rand_proj_ma,
                                                cr_tp=args.cr_tp,
                                                l1=args.l1,
                                                l2=args.l2,
                                                beta=beta,
                                                steer_vs=steer_vs,
                                                steer_tp=args.steer_tp,)

            tar_acc = result["a_cr_ap"]
            if tar_acc > max_tar_acc:
                max_tar_acc = tar_acc
                result["beta"] = beta
                all_result[data_tp] = result

    with open(sd_result + f"ap_result_{args.llm_tp}_{args.input_tp}_atti_{args.atti}_{args.steer_tp}.json", 'w') as fout:
        json.dump(all_result, fout)
        
