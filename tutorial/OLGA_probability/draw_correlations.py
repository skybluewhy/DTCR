import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from cal_KL_div import calculate_kl_divergence


def compute_cdf(data):
    unique_values, counts = np.unique(data, return_counts=True)
    cdf = np.cumsum(counts) / len(data)
    return unique_values, cdf


def get_model_probs(filename):
    all_probs = []
    dat = pd.read_csv("./data/" + filename + "/output.txt")
    for i in range(len(dat)):
        prob = dat["prob"][i]
        if np.log10(prob) < -18 or np.log10(prob) > -5:
            continue
        all_probs.append(prob)
    all_probs = np.array(all_probs)
    return all_probs


data1 = np.log10(get_model_probs("DTCR_blosum"))
data2 = np.log10(get_model_probs("DTCR_random"))
data3 = np.log10(get_model_probs("DTCR_mask"))
data4 = np.log10(get_model_probs("TCR_translate"))
data5 = np.log10(get_model_probs("GRATCR"))
data0 = np.log10(get_model_probs("true_seqs"))
data0_list = list(data0)
cnt = 0
for d in data0_list:
    if d >= -18 and d <= -5:
        cnt += 1
print(cnt / len(data0_list))

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
DTCR_kl = round(calculate_kl_divergence(data1, data0, np.arange(-18, -4, 1)), 3)
DTCR_R_kl = round(calculate_kl_divergence(data2, data0, np.arange(-18, -4, 1)), 3)
DTCR_M_kl = round(calculate_kl_divergence(data3, data0, np.arange(-18, -4, 1)), 3)
TCR_TRANSLATE_kl = round(calculate_kl_divergence(data4, data0, np.arange(-18, -4, 1)), 3)
GRATCR_kl = round(calculate_kl_divergence(data5, data0, np.arange(-18, -4, 1)), 3)
labels = ["Real Binding TCRs", 'DTCR=' + str(DTCR_kl), 'DTCR-R=' + str(DTCR_R_kl),
          'DTCR-M=' + str(DTCR_M_kl), 'TCR-TRANSLATE=' + str(TCR_TRANSLATE_kl), 'GRATCR=' + str(GRATCR_kl)]
plt.figure(figsize=(12, 8))

plt.plot(true_sorted, true_cdf, linestyle='-', color=colors[0], label=labels[0], linewidth=3)
plt.plot(generated_sorted1, generated_cdf1, linestyle='-', color=colors[1], label=labels[1], linewidth=3)
plt.plot(generated_sorted2, generated_cdf2, linestyle='-', color=colors[2], label=labels[2], linewidth=3)
plt.plot(generated_sorted3, generated_cdf3, linestyle='-', color=colors[3], label=labels[3], linewidth=3)
plt.plot(generated_sorted4, generated_cdf4, linestyle='-', color=colors[4], label=labels[4], linewidth=3)
plt.plot(generated_sorted5, generated_cdf5, linestyle='-', color=colors[5], label=labels[5], linewidth=3)

plt.xlabel('', fontsize=18)  # OLGA Generation Probability ($\log_{10}$)
plt.ylabel('', fontsize=18)  # Cumulative Probability
plt.title('', fontsize=18)  # Comparative CDF of OLGA Generation Probabilities
plt.legend(loc='upper left', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.tight_layout()
plt.savefig("./CDF.svg")
