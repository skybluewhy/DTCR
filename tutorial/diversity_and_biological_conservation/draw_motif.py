import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import logomaker as lm
import pandas as pd
from collections import defaultdict


def get_tcr_seqs(filename, target_epi, target_len=16):
    all_tcrs = set()
    data = pd.read_csv("./data/" + filename + "/results.txt")
    for i in range(len(data)):
        epi = data["MT_pep"][i]
        tcr = data["CDR3"][i]
        bind = data["Label"][i]
        if bind == 0 or epi != target_epi:
            continue
        if len(tcr) != target_len:
            continue
        all_tcrs.add(tcr)
    return all_tcrs


def draw_seq_logo(all_tcrs, model, target_len, target_epi, model_name):
    tcr_list = list(all_tcrs)
    position_freq = defaultdict(lambda: defaultdict(int))

    for tcr in tcr_list:
        for pos, amino_acid in enumerate(tcr):
            position_freq[pos][amino_acid] += 1

    total_tcrs = len(tcr_list)
    frequency_data = defaultdict(dict)

    for pos, amino_acid_dict in position_freq.items():
        for amino_acid, count in amino_acid_dict.items():
            frequency_data[pos][amino_acid] = count / total_tcrs

    logo_data = pd.DataFrame(frequency_data).fillna(0).T
    logo_data = logo_data.reindex(sorted(logo_data.columns), axis=1)

    lm.Logo(logo_data, color_scheme='dmslogo_funcgroup', fade_below=0.05, show_spines=False)

    plt.xlabel('', fontsize=16)
    plt.ylabel(model_name, fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("graphs/" + model + "/" + str(target_len) + "/" + target_epi + ".svg")


target_len = 16
target_epi_list = ['LPRRSGAAGA', 'YVLDHLIVV', 'LLLDRLNQL', 'CRVLCCYVL', 'ELAGIGILTV', 'TTDPSFLGRY', 'TPRVTGGGAM',
           'SPRWYFYYL', 'GLCTLVAML', 'RAKFKQLL', 'EAAGIGILTV', 'YLQPRTFLL', 'IVTDFSVIK', 'GILGFVFTL', 'AVFDRKSDAK',
           'LVVDFSQFSR', 'KLGGALQAK', 'LLWNGPMAV', 'STLPETAVVRR', 'NLVPMVATV']
models = ["Real", "DTCR_blosum", "TCR-TRANSLATE", "GRATCR", "DTCR_random", "DTCR_mask"]
model_name = ["Real", "DTCR", "TCR-TRANSLATE", "GRATCR", "DTCR-R", "DTCR-M"]
for model_cnt in range(len(models)):
    model = models[model_cnt]
    for cnt in range(len(target_epi_list)):
        target_epi = target_epi_list[cnt]
        real_tcrs = get_tcr_seqs(model, target_epi, target_len)
        draw_seq_logo(real_tcrs, model, target_len, target_epi, model_name[model_cnt])
        print(model + " " + target_epi)
