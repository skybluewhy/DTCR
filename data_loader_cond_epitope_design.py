from torch.utils.data import Dataset
import pandas as pd
import json
import random


with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
    vocab = json.load(f)


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
            if i > 100:
                break
            tcr = dat["CDR3"][i]
            epitope = dat["MT_pep"][i]
            hla = dat["HLA_sequence"][i]
            label = dat["Label"][i]
            if tcr[0] != "C" or tcr[-1] != "F":  # filter not valid TCR
                continue
            if label == 0:  # filter not bind pairs
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
                target_mask.append(0)  # do not introduce mutation to epitope
            for h_index in range(max_hla_len):
                if h_index < len(hla):
                    input_id.append(vocab[hla[h_index]])
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    attention_mask.append(0)
                target_mask.append(0)  # do not introduce mutation to HLA
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
                        target_mask.append(0)  # head and tail should not mutation
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


class mydataset_infer(Dataset):
    def __init__(self, filename, pep_filename, target_epitope="", seed_num=20, max_tcr_len=30, max_epitope_len=11, max_hla_len=34):
        self.all_input_ids = []
        self.all_attention_mask = []
        self.all_target_mask = []
        self.all_sample_input_ids = []

        dat = pd.read_csv(pep_filename, delimiter=",")
        epi_hla = set()
        for i in range(len(dat)):
            epitope = dat["Epitope"][i]
            hla = dat["MHC A"][i]
            if target_epitope == epitope:  # find the MHC that can be bind to target epitope
                epi_hla.add(epitope + " " + hla)
        epi_hla = list(epi_hla)

        dat = pd.read_csv(filename, delimiter=",")
        seed_tcrs = []
        # sample seed_num * 10 to prevent data insufficiency caused by filtering
        random_n = random.sample(range(len(dat)), seed_num * 10)
        # sample seed tcrs, the seed tcrs' important position can be set as fixed by target_mask (0)
        # In this study, we only fix the head and tail amino acid of seed tcrs
        for n in range(len(random_n)):
            seed_tcrs.append(dat["junction_aa"][random_n[n]])
        for i in range(seed_num * 10):
            tcr = seed_tcrs[i]
            input_id = []
            sample_input_id = []
            attention_mask = []
            target_mask = []
            if tcr[0] != "C" or tcr[-1] != "F" or len(tcr) > max_tcr_len:  # filter not valid tcrs
                continue
            if len(self.all_input_ids) == seed_num:  # break when the seed TCRs' number meets the requirement
                break
            if len(tcr) > 20 or len(tcr) < 10:  # restrict the length of generated TCRs
                continue

            # input_id is the condition (pMHC) + real seed tcrs
            # sample_input_id is the mask (pMHC) + random amino acid sequence (seed tcrs)
            epitope = target_epitope
            hla = random.choice(epi_hla).split(" ")[1]  # randomly select a MHC that can bind to target epitope
            for l in range(max_epitope_len):
                if l < len(epitope):
                    input_id.append(vocab[epitope[l]])
                    sample_input_id.append(21)
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    sample_input_id.append(21)
                    attention_mask.append(0)
                target_mask.append(0)
            for l in range(max_hla_len):
                if l < len(hla):
                    input_id.append(vocab[hla[l]])
                    sample_input_id.append(21)
                    attention_mask.append(1)
                else:
                    input_id.append(0)
                    sample_input_id.append(21)
                    attention_mask.append(0)
                target_mask.append(0)

            for k in range(max_tcr_len):
                if k < min(max_tcr_len, len(tcr)):
                    input_id.append(vocab[tcr[k]])
                    sample_input_id.append(random.randint(1, 20))
                    attention_mask.append(1)
                    if k == 0 or k == len(tcr) - 1:
                        target_mask.append(0)  # head and tail should not mutation
                    else:
                        target_mask.append(1)
                else:
                    input_id.append(0)
                    sample_input_id.append(21)
                    attention_mask.append(0)
                    target_mask.append(0)

            self.all_input_ids.append(input_id)
            self.all_sample_input_ids.append(sample_input_id)
            self.all_attention_mask.append(attention_mask)
            self.all_target_mask.append(target_mask)
        self.cnt = len(self.all_input_ids)

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        return self.all_input_ids[index], self.all_attention_mask[index], self.all_target_mask[index], self.all_sample_input_ids[index]
