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
    f = open("./outputs/" + model_name, 'r')
    for line in f:
        line = line.replace("\n", "")
        line_vec = line.split("\t")
        if float(line_vec[11]) < 28 or float(line_vec[11]) > 33:
            continue
        all_seq_score_list.append(float(line_vec[11]))
        min_n = min(min_n, float(line_vec[11]))
        max_n = max(max_n, float(line_vec[11]))
    all_seq_score_list = np.array(all_seq_score_list)
    return all_seq_score_list, min_n, max_n


def get_model_bit_scores_true(min_n, max_n):
    all_seq_score_list = []
    for i in range(4):
        f = open("./outputs/true" + str(i + 1), 'r')
        for line in f:
            line = line.replace("\n", "")
            line_vec = line.split("\t")
            if float(line_vec[11]) < 28 or float(line_vec[11]) > 33:
                continue
            all_seq_score_list.append(float(line_vec[11]))
            min_n = min(min_n, float(line_vec[11]))
            max_n = max(max_n, float(line_vec[11]))
    all_seq_score_list = np.array(all_seq_score_list)
    return all_seq_score_list, min_n, max_n


def compute_cdf(data):
    unique_values, counts = np.unique(data, return_counts=True)
    cdf = np.cumsum(counts) / len(data)
    return unique_values, cdf


data1, min_n, max_n = get_model_bit_scores("DTCR_blosum", min_n, max_n)
data2, min_n, max_n = get_model_bit_scores("DTCR_random", min_n, max_n)
data3, min_n, max_n = get_model_bit_scores("DTCR_mask", min_n, max_n)
data4, min_n, max_n = get_model_bit_scores("tcr_translate", min_n, max_n)
data5, min_n, max_n = get_model_bit_scores("GRATCR", min_n, max_n)
data0, min_n, max_n = get_model_bit_scores_true(min_n, max_n)

true_sorted, true_cdf = compute_cdf(data0)
generated_sorted1, generated_cdf1 = compute_cdf(data1)
generated_sorted2, generated_cdf2 = compute_cdf(data2)
generated_sorted3, generated_cdf3 = compute_cdf(data3)
generated_sorted4, generated_cdf4 = compute_cdf(data4)
generated_sorted5, generated_cdf5 = compute_cdf(data5)

colors = [
    '#4E4E4E',
    '#D94D4D',
    '#E1812C',
    '#3BA66B',
    '#3274A1',
    '#9C7BCA',
]
DTCR_kl = round(calculate_kl_divergence(data1, data0), 3)
DTCR_R_kl = round(calculate_kl_divergence(data2, data0), 3)
DTCR_M_kl = round(calculate_kl_divergence(data3, data0), 3)
TCR_TRANSLATE_kl = round(calculate_kl_divergence(data4, data0), 3)
GRATCR_kl = round(calculate_kl_divergence(data5, data0), 3)
labels = ["Real Binding TCRs", 'DTCR=' + str(DTCR_kl), 'DTCR-R=' + str(DTCR_R_kl),
          'DTCR-M=' + str(DTCR_M_kl), 'TCR-TRANSLATE=' + str(TCR_TRANSLATE_kl), 'GRATCR=' + str(GRATCR_kl)]

plt.figure(figsize=(12, 8))

plt.plot(true_sorted, true_cdf, linestyle='-', color=colors[0], label=labels[0], linewidth=3)
plt.plot(generated_sorted1, generated_cdf1, linestyle='-', color=colors[1], label=labels[1], linewidth=3)
plt.plot(generated_sorted2, generated_cdf2, linestyle='-', color=colors[2], label=labels[2], linewidth=3)
plt.plot(generated_sorted3, generated_cdf3, linestyle='-', color=colors[3], label=labels[3], linewidth=3)
plt.plot(generated_sorted4, generated_cdf4, linestyle='-', color=colors[4], label=labels[4], linewidth=3)
plt.plot(generated_sorted5, generated_cdf5, linestyle='-', color=colors[5], label=labels[5], linewidth=3)

plt.xlabel('', fontsize=18)  # Blast Bit Score
plt.ylabel('', fontsize=18)  # Cumulative Probability
plt.title('', fontsize=18)  # Comparative CDF of Blast Bit Scores
plt.legend(loc='upper left', fontsize=18)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid(True)
plt.tight_layout()
plt.savefig("./CDF.svg")

