import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from cal_KL_div import calculate_kl_divergence
min_n = 100
max_n = 0
def get_model_bit_scores(model_name, min_n, max_n):
    all_seq_score_list = []
    all_list_left = []
    all_list_right = []
    cnt = 0
    f = open("./outputs/" + model_name, 'r')
    for line in f:
        line = line.replace("\n", "")
        line_vec = line.split("\t")
        cnt += 1
        if float(line_vec[11]) < 28:
            all_list_left.append(float(line_vec[11]))
        elif float(line_vec[11]) > 33:
            all_list_right.append(float(line_vec[11]))
        if float(line_vec[11]) < 28 or float(line_vec[11]) > 33:
            continue
        all_seq_score_list.append(float(line_vec[11]))
        min_n = min(min_n, float(line_vec[11]))
        max_n = max(max_n, float(line_vec[11]))
    return all_seq_score_list, min_n, max_n


def get_model_bit_scores_true(min_n, max_n):
    all_seq_score_list = []
    all_list_left = []
    all_list_right = []
    cnt = 0
    for i in range(4):
        f = open("./outputs/true" + str(i + 1), 'r')
        for line in f:
            line = line.replace("\n", "")
            line_vec = line.split("\t")
            cnt += 1
            if float(line_vec[11]) < 28:
                all_list_left.append(float(line_vec[11]))
            elif float(line_vec[11]) > 33:
                all_list_right.append(float(line_vec[11]))
            if float(line_vec[11]) < 28 or float(line_vec[11]) > 33:
                continue
            all_seq_score_list.append(float(line_vec[11]))
            min_n = min(min_n, float(line_vec[11]))
            max_n = max(max_n, float(line_vec[11]))
    return all_seq_score_list, min_n, max_n


data1, min_n, max_n = get_model_bit_scores("DTCR_blosum", min_n, max_n)
data2, min_n, max_n = get_model_bit_scores("DTCR_random", min_n, max_n)
data3, min_n, max_n = get_model_bit_scores("DTCR_mask", min_n, max_n)
data4, min_n, max_n = get_model_bit_scores("tcr_translate", min_n, max_n)
data5, min_n, max_n = get_model_bit_scores("GRATCR", min_n, max_n)
data0, min_n, max_n = get_model_bit_scores_true(min_n, max_n)

bins = np.arange(28, 33, 0.5)

hist1, bin_edges1 = np.histogram(data1, bins=bins, density=True)
hist2, bin_edges2 = np.histogram(data2, bins=bins, density=True)
hist3, bin_edges3 = np.histogram(data3, bins=bins, density=True)
hist4, bin_edges4 = np.histogram(data4, bins=bins, density=True)
hist5, bin_edges5 = np.histogram(data5, bins=bins, density=True)
hist0, bin_edges0 = np.histogram(data0, bins=bins, density=True)
hist0 = hist0/2
hist1 = hist1/2
hist2 = hist2/2
hist3 = hist3/2
hist4 = hist4/2
hist5 = hist5/2

colors = [
    '#4E4E4E',
    '#D94D4D',
    '#E1812C',
    '#3BA66B',
    '#3274A1',
    '#9C7BCA',
]
labels = ["Real Binding TCRs", 'DTCR', 'DTCR-R', 'DTCR-M', 'TCR-TRANSLATE', 'GRATCR']

bin_centers = (bin_edges1[:-1] + bin_edges1[1:]) / 2

plt.figure(figsize=(12, 6))

plt.plot(bin_centers, hist0, color=colors[0], marker='o', linestyle='-', linewidth=2, markersize=6, label=labels[0])
plt.plot(bin_centers, hist1, color=colors[1], marker='s', linestyle='-', linewidth=2, markersize=6, label=labels[1])
plt.plot(bin_centers, hist2, color=colors[2], marker='^', linestyle='-', linewidth=2, markersize=6, label=labels[2])
plt.plot(bin_centers, hist3, color=colors[3], marker='d', linestyle='-', linewidth=2, markersize=6, label=labels[3])
plt.plot(bin_centers, hist4, color=colors[4], marker='*', linestyle='-', linewidth=2, markersize=8, label=labels[4])
plt.plot(bin_centers, hist5, color=colors[5], marker='p', linestyle='-', linewidth=2, markersize=8, label=labels[5])

plt.xlabel('', fontsize=12)  # Blast Bit Score
plt.ylabel('', fontsize=12)  # Density
plt.title('', fontsize=14)  # Distribution Comparison on Blast Bit Scores
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=16, bbox_to_anchor=(0.5, 1))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("./bit_score_line_graph.svg")
