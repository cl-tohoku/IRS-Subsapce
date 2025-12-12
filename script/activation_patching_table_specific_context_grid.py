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


def load_tb_data(tokenizer, num_samples, data_file, rand=True,):
    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if rand:
        random.shuffle(data)

    que_temp = "Based on the context, given like relation {ENT1} to {ATT1} , {ENT2} to"
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

    prompts_story1_1 = []
    prompts_story2_1 = []
    prompts_story3_1 = []
    prompts_story4_1 = []

    prompts_story1_2 = []
    prompts_story2_2 = []
    prompts_story3_2 = []
    prompts_story4_2 = []

    prompts_story1_3 = []
    prompts_story2_3 = []
    prompts_story3_3 = []
    prompts_story4_3 = []

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

        que1_1 = que_temp.replace("{ENT1}", ent2).replace("{ATT1}", att1_2).replace("{ENT2}", ent1)
        que2_1 = que_temp.replace("{ENT1}", ent2).replace("{ATT1}", att2_2).replace("{ENT2}", ent1)
        que3_1 = que_temp.replace("{ENT1}", ent2).replace("{ATT1}", att3_2).replace("{ENT2}", ent1)
        que4_1 = que_temp.replace("{ENT1}", ent2).replace("{ATT1}", att4_2).replace("{ENT2}", ent1)

        que1_2 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1_1).replace("{ENT2}", ent2)
        que2_2 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att2_1).replace("{ENT2}", ent2)
        que3_2 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att3_1).replace("{ENT2}", ent2)
        que4_2 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att4_1).replace("{ENT2}", ent2)

        que1_3 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att1_1).replace("{ENT2}", ent3)
        que2_3 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att2_1).replace("{ENT2}", ent3)
        que3_3 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att3_1).replace("{ENT2}", ent3)
        que4_3 = que_temp.replace("{ENT1}", ent1).replace("{ATT1}", att4_1).replace("{ENT2}", ent3)
        
        prompts_table.append(ctx_t)
        prompts_temp.append(ctx)
        prompts_story.append(ctx_s)
        
        prompts_story1_1.append(ctx_s + que1_1)
        prompts_story2_1.append(ctx_s + que2_1)
        prompts_story3_1.append(ctx_s + que3_1)
        prompts_story4_1.append(ctx_s + que4_1)

        prompts_story1_2.append(ctx_s + que1_2)
        prompts_story2_2.append(ctx_s + que2_2)
        prompts_story3_2.append(ctx_s + que3_2)
        prompts_story4_2.append(ctx_s + que4_2)

        prompts_story1_3.append(ctx_s + que1_3)
        prompts_story2_3.append(ctx_s + que2_3)
        prompts_story3_3.append(ctx_s + que3_3)
        prompts_story4_3.append(ctx_s + que4_3)
        
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

    input_tokens_story = tokenizer(prompts_story1_1, padding=True, return_tensors="pt")
    input_ids_story1_1 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story2_1, padding=True, return_tensors="pt")
    input_ids_story2_1 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story3_1, padding=True, return_tensors="pt")
    input_ids_story3_1 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story4_1, padding=True, return_tensors="pt")
    input_ids_story4_1 = input_tokens_story["input_ids"]

    input_tokens_story = tokenizer(prompts_story1_2, padding=True, return_tensors="pt")
    input_ids_story1_2 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story2_2, padding=True, return_tensors="pt")
    input_ids_story2_2 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story3_2, padding=True, return_tensors="pt")
    input_ids_story3_2 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story4_2, padding=True, return_tensors="pt")
    input_ids_story4_2 = input_tokens_story["input_ids"]

    input_tokens_story = tokenizer(prompts_story1_3, padding=True, return_tensors="pt")
    input_ids_story1_3 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story2_3, padding=True, return_tensors="pt")
    input_ids_story2_3 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story3_3, padding=True, return_tensors="pt")
    input_ids_story3_3 = input_tokens_story["input_ids"]
    input_tokens_story = tokenizer(prompts_story4_3, padding=True, return_tensors="pt")
    input_ids_story4_3 = input_tokens_story["input_ids"]
    
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
            labels_x, labels_y, labels_z,
            input_ids_story1_1, input_ids_story2_1, input_ids_story3_1, input_ids_story4_1,
            input_ids_story1_2,	input_ids_story2_2, input_ids_story3_2, input_ids_story4_2,
            input_ids_story1_3,	input_ids_story2_3, input_ids_story3_3, input_ids_story4_3,)


def load_dataloader(
        tokenizer: AutoTokenizer,
        data_file: str,
        num_samples: int,
        batch_size: int,):
    
    raw_data = load_tb_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=data_file,
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

    base_tokens_story1_1 = raw_data[21]
    base_tokens_story2_1 = raw_data[22]
    base_tokens_story3_1 = raw_data[23]
    base_tokens_story4_1 = raw_data[24]

    base_tokens_story1_2 = raw_data[25]
    base_tokens_story2_2 = raw_data[26]
    base_tokens_story3_2 = raw_data[27]
    base_tokens_story4_2 = raw_data[28]

    base_tokens_story1_3 = raw_data[29]
    base_tokens_story2_3 = raw_data[30]
    base_tokens_story3_3 = raw_data[31]
    base_tokens_story4_3 = raw_data[32]
    
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
            "base_tokens_story1_1": base_tokens_story1_1,
            "base_tokens_story2_1": base_tokens_story2_1,
            "base_tokens_story3_1": base_tokens_story3_1,
            "base_tokens_story4_1": base_tokens_story4_1,
            "base_tokens_story1_2": base_tokens_story1_2,
            "base_tokens_story2_2": base_tokens_story2_2,
            "base_tokens_story3_2": base_tokens_story3_2,
            "base_tokens_story4_2": base_tokens_story4_2,
            "base_tokens_story1_3": base_tokens_story1_3,
            "base_tokens_story2_3": base_tokens_story2_3,
            "base_tokens_story3_3": base_tokens_story3_3,
            "base_tokens_story4_3": base_tokens_story4_3,
        }
    ).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def activation_patching_resid_ap(
        inputs: tuple=None,
        output: torch.Tensor=None,
        layer: str=None,
        model: AutoModelForCausalLM=None,
        input_tokens: list=None,
        input_tp: str="story",
        index: str="2_1",
        proj_ma: torch.tensor=None,
        beta: float=0.55,
        alpha: float=1.0,
        grid: torch.tensor=None,
        ):
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(output, tuple):
        outputs = output[0]

    outputs = rearrange(
        outputs,
        "batch seq_len d_resid -> batch seq_len d_resid",
        d_resid=model.config.hidden_size,
    )
    
    layer_index = int(layer.split(".")[2])
    
    for batch in range(outputs.size(0)):
        pos = compute_prev_clue_pos(
            input_tokens[f"base_tokens_{input_tp}{index}"][batch],
            input_tokens[f"atts{index}"][batch],
        )
        emb = outputs[batch, pos]
        p_emb = torch.matmul(emb, proj_ma)
        steer = torch.matmul(grid - p_emb, proj_ma.T)
        emb_cr = alpha * emb + beta * steer
        outputs[batch, pos] = emb_cr
        #outputs[batch, pos] = 0
        
    output = rearrange(
        outputs,
        "batch seq_len d_resid -> batch seq_len d_resid",
        d_resid=model.config.hidden_size,
    )
    torch.cuda.empty_cache()
    return (output,)


def eval_model_performance(model, dataloader, input_tp="story", index="1_2"):
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

            outputs = model(input_ids=output[f"base_tokens_{input_tp}{index}"])

            for bi in range(output[f"atts{index}"].size(0)):
                label = output[f"atts{index}"][bi].item()
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
        input_tp: str="story",
        proj_ma: torch.tensor=None,
        beta: float=0.55,
        alpha: float=1.0,
        index: str="1_2",
        grids: torch.tensor=None,
        nb_loop: int=1,):
    print(beta)
    accs = defaultdict(list)
    apply_softmax = torch.nn.Softmax(dim=-1)
    acc_cri = 0.1
    for loop_idx in range(nb_loop):
        acc = 0.0
        while acc < acc_cri:
            dataloader = load_dataloader(
                tokenizer=tokenizer,
                data_file=data_file,
                num_samples=num_sample,
                batch_size=batch_size,
            )
            acc, logit = eval_model_performance(model, dataloader, input_tp=input_tp, index=index)
        print(f"Model Acc.: {acc} Logit: {logit}")
        hook_points = [
            f"model.layers.{layer}" for layer in range(l1, l2)
        ]
        
        for grid in tqdm(grids):
            correct_count, total_count = 0, 0
            total_logit = []

            for bi, inputs in enumerate(dataloader):
                with TraceDict(
                        model,
                        hook_points,
                        retain_input=True,
                        edit_output=partial(
                            activation_patching_resid_ap,
                            model=model,
                            input_tokens=inputs,
                            input_tp=input_tp,
                            proj_ma=proj_ma,
                            beta=beta,
                            alpha=alpha,
                            grid=grid,
                            index=index,
                        ),
                ) as _:
                    outputs = model(inputs[f"base_tokens_{input_tp}{index}"].to(model.device))
                    
                for idx in range(inputs[f"base_tokens_{input_tp}{index}"].size(0)):
                    label = inputs[f"atts{index}"][idx].item()
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
            accs[grid].append((acc, round(m_logit, 2)))

    result = {}
    for grid, lst_acc in accs.items():
        lst_acc = np.array(lst_acc)
        acc_m = np.mean(lst_acc[:,0])
        acc_std = np.std(lst_acc[:,0])
        logit_m = np.mean(lst_acc[:,1])
        logit_std = np.std(lst_acc[:,1])
        
        print(f"Accuracy: {acc_m} ({acc_std}) , Logit: {logit_m} ({logit_std})")
        result[str(grid)] = (acc_m, acc_std, logit_m, logit_std)
        
    return result


###Learning Projection Matrix Part###

import torch

def pls_fit_projection(X, Y, n_components=2, tol=1e-6, max_iter=500):
    """
    Learn PLS projection matrices:
      - X -> latent subspace (P_X)
      - X -> Y (B)
    """
    # Center data
    X_mean = X.mean(0, keepdim=True)
    Y_mean = Y.mean(0, keepdim=True)
    Xc = X - X_mean
    Yc = Y - Y_mean
    
    n, v = Xc.shape
    _, d = Yc.shape
    
    device = X.device
    W = torch.zeros(v, n_components, device=device)
    P = torch.zeros(v, n_components, device=device)
    Q = torch.zeros(d, n_components, device=device)
    
    X_res, Y_res = Xc.clone(), Yc.clone()
    
    for a in range(n_components):
        u = Y_res[:, [0]]  # initialize u from Y

        for _ in range(max_iter):
            w = X_res.T @ u
            w = w / torch.norm(w)
            t = X_res @ w
            q = Y_res.T @ t
            q = q / torch.norm(q)
            u_new = Y_res @ q

            if torch.norm(u_new - u) < tol:
                break
            u = u_new

        p = X_res.T @ t / (t.T @ t)
        q = Y_res.T @ t / (t.T @ t)

        W[:, a] = w.squeeze()
        P[:, a] = p.squeeze()
        Q[:, a] = q.squeeze()

        # Deflate
        X_res -= t @ p.T
        Y_res -= t @ q.T

    # Compute projection matrices
    inv_term = torch.inverse(P.T @ W)
    P_X = W @ inv_term              # X → subspace
    B = P_X @ Q.T                   # X → Y

    return P_X, X_mean, Y_mean


def sample_even_points(X, m, rate=0.15):
    """
    Evenly sample m points in the N-D space of X using Latin Hypercube Sampling (LHS),
    fully in PyTorch and on the same device as X.

    Args:
        X: (n, d) torch.Tensor
        m: number of samples to generate
    Returns:
        samples: (m, d) torch.Tensor, evenly distributed across X's range
    """
    device = X.device
    n, d = X.shape

    # Compute per-dimension min and max
    mins, _ = X.min(dim=0)
    maxs, _ = X.max(dim=0)

    # LHS in [0, 1]
    # Each dimension is split into m bins, one point per bin
    seg_edges = torch.linspace(0, 1, m + 1, device=device)
    lower, upper = seg_edges[:-1], seg_edges[1:]

    # Generate one sample per bin per dimension
    r = torch.rand((m, d), device=device) * (upper - lower).unsqueeze(1) + lower.unsqueeze(1)

    # Randomly permute bins in each dimension
    for j in range(d):
        r[:, j] = r[torch.randperm(m, device=device), j]

    # Scale to the actual data range
    if rate == 0.0:
        samples = mins + r * (maxs - mins)
    else:
        rn = maxs - mins
        nmaxs = maxs - rate * rn
        nmins = mins + rate * rn
        samples = nmins + r * (nmaxs - nmins)
                                            
    return samples


def learn_proj_ma(
        model: AutoModelForCausalLM,
        dataloader: torch.utils.data.DataLoader,
        layer: int=15,
        n_components: int=5,
        n_grids: int=100,
        input_tp: str="story"):
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

    X = torch.cat([xs_atts1_1, xs_atts1_2, xs_atts1_3,
                   xs_atts2_1, xs_atts2_2, xs_atts2_3,
                   xs_atts3_1, xs_atts3_2, xs_atts3_3,
                   xs_atts4_1, xs_atts4_2, xs_atts4_3,], dim=0)

    n11 = xs_atts1_1.size(0)
    n12 = xs_atts1_2.size(0)
    n13 = xs_atts1_3.size(0)
    n21 = xs_atts2_1.size(0)
    n22 = xs_atts2_2.size(0)
    n23 = xs_atts2_3.size(0)
    n31 = xs_atts3_1.size(0)
    n32 = xs_atts3_2.size(0)
    n33 = xs_atts3_3.size(0)
    n41 = xs_atts4_1.size(0)
    n42 = xs_atts4_2.size(0)
    n43 = xs_atts4_3.size(0)
    y11 = torch.tensor([[1., 1.]]).repeat(n11, 1)
    y12 = torch.tensor([[1., 2.]]).repeat(n12, 1)
    y13 = torch.tensor([[1., 3.]]).repeat(n13, 1)
    y21 = torch.tensor([[2., 1.]]).repeat(n21, 1)
    y22 = torch.tensor([[2., 2.]]).repeat(n22, 1)
    y23 = torch.tensor([[2., 3.]]).repeat(n23, 1)
    y31 = torch.tensor([[3., 1.]]).repeat(n31, 1)
    y32 = torch.tensor([[3., 2.]]).repeat(n32, 1)
    y33 = torch.tensor([[3., 3.]]).repeat(n33, 1)
    y41 = torch.tensor([[4., 1.]]).repeat(n41, 1)
    y42 = torch.tensor([[4., 2.]]).repeat(n42, 1)
    y43 = torch.tensor([[4., 3.]]).repeat(n43, 1)

    Y = torch.cat([y11, y12, y13,
                   y21, y22, y23,
                   y31, y32, y33,
                   y41, y42, y43,], dim=0)

    proj_ma, _, _ = pls_fit_projection(X, Y, n_components=n_components)
    p_X = torch.matmul(X, proj_ma)
    grids = sample_even_points(p_X, m=n_grids)
    grids = grids.to(torch.bfloat16).to(model.device)
    proj_ma = proj_ma.to(torch.bfloat16).to(model.device)
    return proj_ma, grids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for activation patching")
    
    parser.add_argument("--llm_tp", type=str, default="llama", help="llama / qwen")
    parser.add_argument("--input_tp", type=str, default="story", help="table / temp / story")
    parser.add_argument("--cr_tp", type=str, default="acr", help="acr / ecr")
    parser.add_argument("--index", type=str, default="1_2", help="index in the space")

    parser.add_argument("--l1", type=int, default=10, help="0 ~ 31")
    parser.add_argument("--l2", type=int, default=20, help="0 ~ 31")
    parser.add_argument("--num_sample", type=int, default=50, help="")
    parser.add_argument("--batch_size", type=int, default=50, help="")
    parser.add_argument("--nb_loop", type=int, default=1, help="")
    parser.add_argument("--beta", type=float, default=0.4, help="job/create:0.4, others:0.4")
    parser.add_argument("--alpha", type=float, default=1.0, help="")
    parser.add_argument("--n_grids", type=int, default=10000, help="10000")
    parser.add_argument("--n_components", type=int, default=2, help="")
    parser.add_argument("--proj_layer", type=int, default=15, help="")
    parser.add_argument("--learn_proj_ma", action="store_true", help="Learn proj ma or not (default: False)")
    args = parser.parse_args()
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(args.llm_tp, device)
    print("Model and Tokenizer loaded")

    data_tps = ["job", "create", "space", "relation", "city"]
    #data_tps = ["space", "city"]
    sd_result = "./data/data_table_result_specific_context_grid/"

    for data_tp in data_tps:
        sd = "./data/data_table/"
        sf_data = sd + f"{data_tp}_tts_all.jsonl"

        if args.learn_proj_ma:
            dataloader = load_dataloader(tokenizer=tokenizer,
                                         data_file=sf_data,
                                         num_samples=1000,
                                         batch_size=100,)
            proj_ma, grids = learn_proj_ma(
                model=model,
                dataloader=dataloader,
                layer=args.proj_layer,
                n_components=args.n_components,
                n_grids=args.n_grids,
                input_tp="story")

            torch.save(proj_ma, sd_result + f"{args.llm_tp}_{data_tp}_proj_ma.pt")
            torch.save(grids, sd_result + f"{args.llm_tp}_{data_tp}_grids.pt")
        else:
            proj_ma = torch.load(sd_result + f"{args.llm_tp}_{data_tp}_proj_ma.pt").to(model.device)
            grids = torch.load(sd_result + f"{args.llm_tp}_{data_tp}_grids.pt").to(model.device)

            if args.llm_tp == "llama":
                args.alpha = 1.00
                if data_tp in ["job", "create"]:
                    args.beta=-0.4
                else:
                    args.beta=-0.4
            else:
                args.alpha = 1.00
                if data_tp in ["job", "create"]:
                    args.beta=-0.5
                else:
                    args.beta=-0.5
            
            print(f"Patching {data_tp} format {args.input_tp} ...")
            result = act_patching_main_resid_ap(model=model,
                                                tokenizer=tokenizer,
                                                data_file=sf_data,
                                                input_tp=args.input_tp,
                                                nb_loop=args.nb_loop,
                                                num_sample=args.num_sample,
                                                batch_size=args.batch_size,
                                                l1=args.l1,
                                                l2=args.l2,
                                                beta=args.beta,
                                                alpha=args.alpha,
                                                proj_ma=proj_ma,
                                                index=args.index,
                                                grids=grids)
                
            with open(sd_result + f"ap_result_{args.llm_tp}_{data_tp}_{args.index}.json", 'w') as fout:
                json.dump(result, fout)
        
