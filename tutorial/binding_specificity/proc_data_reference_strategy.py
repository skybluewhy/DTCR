import os
import pandas as pd
import random


dat = pd.read_csv("../../data/11M_dataset.csv")
all_ref_tcrs = []
for i in range(len(dat)):
    tcr = dat["junction_aa"][i]
    all_ref_tcrs.append(tcr)


def get_all_pos(filename, epi):
    all_pos = []
    f = open(filename, 'r')
    for line in f:
        line_vec = line.replace("\n", "").split(" ")
        mhc = line_vec[0]
        tcr = line_vec[1]
        all_pos.append(epi + "," + mhc + "," + tcr + ",1")
    return all_pos


def get_all_neg_ref(filename, epi):
    all_neg = []
    f = open(filename, 'r')
    for line in f:
        line_vec = line.replace("\n", "").split(" ")
        mhc = line_vec[0]
        tcr = random.choice(all_ref_tcrs)
        all_neg.append(epi + "," + mhc + "," + tcr + ",0")
    return all_neg


all_epi = ['LPRRSGAAGA', 'YVLDHLIVV', 'LLLDRLNQL', 'CRVLCCYVL', 'ELAGIGILTV', 'TTDPSFLGRY', 'TPRVTGGGAM', 'SPRWYFYYL',
           'GLCTLVAML', 'RAKFKQLL', 'EAAGIGILTV', 'YLQPRTFLL', 'IVTDFSVIK', 'GILGFVFTL', 'AVFDRKSDAK', 'LVVDFSQFSR',
           'KLGGALQAK', 'LLWNGPMAV', 'STLPETAVVRR', 'NLVPMVATV']
all_txt = "MT_pep,HLA_sequence,CDR3,Label\n"
all_epi_pos = []
all_epi_neg = []
for e in all_epi:
    print(e)
    pos_list = get_all_pos("../../generation_results/" + e + ".txt", e)
    selected_neg = get_all_neg_ref("../../generation_results/" + e + ".txt", e)
    all_epi_pos += pos_list
    all_epi_neg += selected_neg
f = open("eval_data_reference_strategy.txt", 'w')
f.write(all_txt + "\n".join(all_epi_pos + all_epi_neg))
f.close()
