from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json
import random
import pickle


with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
    vocab = json.load(f)
alphabets = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}


class mydataset(Dataset):
    def __init__(self, filename, max_tcr_len=30, max_epitope_len=11, max_hla_len=34):
        self.all_cond_input_ids = []
        self.all_cond_attention_mask = []
        self.all_cond_target_mask = []
        self.all_input_ids = []
        self.all_attention_mask = []
        self.all_target_mask = []

        dat = pd.read_csv(filename, delimiter=",")
        for i in range(len(dat)):
            tcr = dat["CDR3"][i]
            epitope = dat["MT_pep"][i]
            hla = dat["HLA_sequence"][i]
            label = dat["Label"][i]
            if tcr[0] != "C" or tcr[-1] != "F":
                continue
            if label == 0:
                continue

            input_id = []
            attention_mask = []
            target_mask = []
            for e_index in range(max_epitope_len):
                if e_index < len(epitope):
                    input_id.append(vocab[epitope[e_index]])
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    attention_mask.append(0)
                target_mask.append(0)
            for h_index in range(max_hla_len):
                if h_index < len(hla):
                    input_id.append(vocab[hla[h_index]])
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    attention_mask.append(0)
                target_mask.append(0)
            self.all_cond_input_ids.append(input_id)
            self.all_cond_attention_mask.append(attention_mask)
            self.all_cond_target_mask.append(target_mask)

            input_id = []
            attention_mask = []
            target_mask = []
            for l in range(max_tcr_len):
                if l < len(tcr):
                    input_id.append(vocab[tcr[l]])
                    attention_mask.append(1)
                    if l == 0 or l == len(tcr) - 1:
                        target_mask.append(0)
                    else:
                        target_mask.append(1)
                else:
                    input_id.append(0)
                    attention_mask.append(0)
                    target_mask.append(0)

            self.all_input_ids.append(input_id)
            self.all_attention_mask.append(attention_mask)
            self.all_target_mask.append(target_mask)
        self.cnt = len(self.all_input_ids)

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        back_input_ids = []
        back_attention_mask = []
        back_target_mask = []
        back_input_ids.append(self.all_cond_input_ids[index] + self.all_input_ids[index])
        back_attention_mask.append(self.all_cond_attention_mask[index] + self.all_attention_mask[index])
        back_target_mask.append(self.all_cond_target_mask[index] + self.all_target_mask[index])
        return back_input_ids, back_attention_mask, back_target_mask


class mydataset_reinforce_learn(Dataset):
    def __init__(self, filename, pep_filename, target_epitope="FVDGVPFVV", seed_num=1000, max_tcr_len=30, max_epitope_len=11, max_hla_len=34, batch_size=64):
        self.all_input_ids = []
        self.all_attention_mask = []
        self.all_target_mask = []
        self.all_sample_input_ids = []

        dat = pd.read_csv(pep_filename, delimiter=",")
        epi_hla = set()
        for i in range(len(dat)):
            epitope = dat["Epitope"][i]
            hla = dat["MHC A"][i]
            if target_epitope == epitope:
                epi_hla.add(epitope + " " + hla)
        epi_hla = list(epi_hla)

        dat = pd.read_csv(filename, delimiter=",")
        seed_tcrs = []
        random_n = random.sample(range(len(dat)), seed_num * 10)
        for n in range(len(random_n)):
            seed_tcrs.append(dat["junction_aa"][random_n[n]])

        for i in range(seed_num * 10):
            tcr = seed_tcrs[i]
            input_id = []
            sample_input_id = []
            attention_mask = []
            target_mask = []
            if tcr[0] != "C" or tcr[-1] != "F" or len(tcr) > max_tcr_len:
                continue
            if len(self.all_input_ids) == seed_num:
                break
            if len(tcr) > 30 or len(tcr) < 10:
                continue

            epitope = target_epitope
            hla = random.choice(epi_hla).split(" ")[1]
            for l in range(max_epitope_len):
                if l < len(epitope):
                    input_id.append(vocab[epitope[l]])
                    sample_input_id.append(vocab[epitope[l]])
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    sample_input_id.append(0)
                    attention_mask.append(0)
                target_mask.append(0)
            for l in range(max_hla_len):
                if l < len(hla):
                    input_id.append(vocab[hla[l]])
                    sample_input_id.append(vocab[hla[l]])
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    sample_input_id.append(0)
                    attention_mask.append(0)
                target_mask.append(0)

            for k in range(max_tcr_len):
                if k < min(max_tcr_len, len(tcr)):

                    input_id.append(vocab[tcr[k]])
                    sample_input_id.append(vocab[tcr[k]])
                    attention_mask.append(1)
                    if k < 1 or k >= len(tcr) - 1:
                        target_mask.append(0)  # 头和尾不能变异
                    else:
                        target_mask.append(1)
                else:
                    input_id.append(0)
                    sample_input_id.append(0)
                    attention_mask.append(0)
                    target_mask.append(0)

            self.all_input_ids += [input_id] * batch_size
            self.all_sample_input_ids += [sample_input_id] * batch_size
            self.all_attention_mask += [attention_mask] * batch_size
            self.all_target_mask += [target_mask] * batch_size
        self.cnt = len(self.all_input_ids)

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        return self.all_input_ids[index], self.all_attention_mask[index], self.all_target_mask[index], self.all_sample_input_ids[index]
