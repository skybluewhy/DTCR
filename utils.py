import torch
import numpy as np


def fori_loop(lower, upper, body_fun, init_val, att, target):
    val = init_val
    for i in range(lower, upper):
        val, att, target = body_fun(i, val, att, target)
    return val


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    return carry, torch.stack(ys) if ys[0] is not None else None


def word_frequency(path, basic_freq=.5):
    freq = torch.load(path)
    return basic_freq + (1 - basic_freq) * freq / freq.mean()


def min_max_norm(t, dim):
    return ((t - t.min(dim=dim, keepdims=True).values) / (t.max(dim=dim, keepdims=True).values - t.min(dim=dim, keepdims=True).values)) * 2 - 1


import json
def proc_data_for_bind(input_ids):
    with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
        token_dict = json.load(f)
    with open("./bert_model/bind_pred_model_vocab.json", 'r', encoding='UTF-8') as f:
        bind_token_dict = json.load(f)
    bind_token_dict["-"] = 0
    bind_token_dict["#"] = 21
    token_dict = dict(zip(token_dict.values(), token_dict.keys()))
    new_input_ids = []
    dim0, dim1 = input_ids.shape
    for i in range(dim0):
        cur_input = []
        for j in range(dim1):
            cur_input_n = input_ids[i][j].item()
            cur_input.append(bind_token_dict[token_dict[cur_input_n]])
        new_input_ids.append(cur_input)
    new_input_ids = torch.tensor(new_input_ids).to(input_ids.device)
    return new_input_ids
