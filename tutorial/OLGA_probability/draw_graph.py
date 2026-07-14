import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd


def get_model_probs(filename):
    all_probs = []
    dat = pd.read_csv("./data/" + filename + "/output.txt")
    for i in range(len(dat)):
        prob = dat["prob"][i]
        all_probs.append(prob)
    all_probs = np.array(all_probs)
    return all_probs


data1 = np.log10(get_model_probs("DTCR_blosum"))
data2 = np.log10(get_model_probs("DTCR_random"))
data3 = np.log10(get_model_probs("DTCR_mask"))
data4 = np.log10(get_model_probs("TCR_translate"))
data5 = np.log10(get_model_probs("GRATCR"))
data0 = np.log10(get_model_probs("true_seqs"))

bins = np.arange(-18, -4, 1)

hist1, _ = np.histogram(data1, bins=bins, density=True)
hist2, _ = np.histogram(data2, bins=bins, density=True)
hist3, _ = np.histogram(data3, bins=bins, density=True)
hist4, _ = np.histogram(data4, bins=bins, density=True)
hist5, _ = np.histogram(data5, bins=bins, density=True)
hist0, _ = np.histogram(data0, bins=bins, density=True)

colors = [
    '#4E4E4E',
    '#D94D4D',
    '#E1812C',
    '#3BA66B',
    '#3274A1',
    '#9C7BCA',
]
labels = ["Real Binding TCRs", 'DTCR', 'DTCR-R', 'DTCR-M', 'TCR-TRANSLATE', 'GRATCR']

plt.figure(figsize=(10, 6))
plt.bar(bins[:-1], hist0, width=0.15, color=colors[0], edgecolor='black', align='edge', label=labels[0])
plt.bar(bins[:-1] + 0.15, hist1, width=0.15, color=colors[1], edgecolor='black', align='edge', label=labels[1])
plt.bar(bins[:-1] + 0.30, hist2, width=0.15, color=colors[2], edgecolor='black', align='edge', label=labels[2])
plt.bar(bins[:-1] + 0.45, hist3, width=0.15, color=colors[3], edgecolor='black', align='edge', label=labels[3])
plt.bar(bins[:-1] + 0.60, hist4, width=0.15, color=colors[4], edgecolor='black', align='edge', label=labels[4])
plt.bar(bins[:-1] + 0.75, hist5, width=0.15, color=colors[5], edgecolor='black', align='edge', label=labels[5])

plt.xlabel('', fontsize=12)  # OLGA Generation Probability ($\log_{10}$)
plt.ylabel('', fontsize=12)  # Density
plt.title('', fontsize=14)  # Distribution Comparison on OLGA Generation Probabilities
plt.xticks(np.arange(-18, -4), fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=16)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("./OLGA_graph.svg")
