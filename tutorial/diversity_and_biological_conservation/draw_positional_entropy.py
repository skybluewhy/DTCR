import pandas as pd
import matplotlib
matplotlib.use("TkAgg")


target_len = 20
dat = pd.read_csv("./data/DTCR_blosum/results.txt")
all_true_tcr = []
min_n = 30
max_n = 0
for i in range(len(dat)):
    epi = dat["MT_pep"][i]
    mhc = dat["HLA_sequence"][i]
    tcr = dat["CDR3"][i]
    label = dat["Label"][i]
    if label == 0:
        continue
    if len(tcr) != target_len:
        continue
    all_true_tcr.append(tcr)
    min_n = min(min_n, len(tcr))
    max_n = max(max_n, len(tcr))
print(min_n)
print(max_n)

dat = pd.read_csv("./data/true_seqs/results.txt")
all_gen_tcr = []
min_n = 30
max_n = 0
for i in range(len(dat)):
    epi = dat["MT_pep"][i]
    mhc = dat["HLA_sequence"][i]
    tcr = dat["CDR3"][i]
    label = dat["Label"][i]
    if len(tcr) != target_len:
        continue
    if label == 0:
        continue
    all_gen_tcr.append(tcr)
    min_n = min(min_n, len(tcr))
    max_n = max(max_n, len(tcr))
print(min_n)
print(max_n)

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import entropy


def calculate_entropy_at_positions(seqs):
    max_length = max(len(seq) for seq in seqs)
    position_entropies = []
    for pos in range(max_length):
        chars_at_pos = [seq[pos] for seq in seqs if len(seq) > pos]
        char_counts = Counter(chars_at_pos)
        pos_entropy = entropy(list(char_counts.values()), base=2)
        position_entropies.append(pos_entropy)
    return position_entropies


true_entropies = calculate_entropy_at_positions(all_true_tcr)
gen_entropies = calculate_entropy_at_positions(all_gen_tcr)
print(sum(true_entropies)/(len(true_entropies)-2))
print(sum(gen_entropies)/(len(gen_entropies)-2))
print("Real max: " + str(max(true_entropies[4:-4])) + " min: " + str(min(true_entropies[4:-4])))
print("Generate max: " + str(max(gen_entropies[4:-4])) + " min: " + str(min(gen_entropies[4:-4])))

print(true_entropies)
print(gen_entropies)
from scipy.stats import pearsonr
correlation, _ = pearsonr(true_entropies, gen_entropies)
print("Pearsonr: " + str(correlation))
from scipy.stats import spearmanr
spearman_correlation, _ = spearmanr(true_entropies, gen_entropies)
print("spearmanr: " + str(spearman_correlation))

max_length = max(len(true_entropies), len(gen_entropies))
true_entropies += [0] * (max_length - len(true_entropies))
gen_entropies += [0] * (max_length - len(gen_entropies))

bar_width = 0.35
index = np.arange(max_length)

plt.figure(figsize=(12, 8))
plt.bar(index, true_entropies, bar_width, label='Real Binding TCRs', color='#4E4E4E', alpha=0.7)
plt.bar(index, [-x for x in gen_entropies], bar_width, label='Generated TCRs', color='#D94D4D', alpha=0.7)

plt.xlabel('Position in Sequence', fontsize=22)
plt.ylabel('Entropy', fontsize=22)
plt.title('Entropy at Different Positions in TCR Sequences', fontsize=22)
plt.xticks(index, [str(i+1) for i in range(max_length)])
plt.legend(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./graphs/entropy" + str(target_len) + ".svg")
