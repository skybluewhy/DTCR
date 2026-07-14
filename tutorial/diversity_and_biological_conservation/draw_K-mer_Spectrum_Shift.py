import pandas as pd
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use("TkAgg")


def kmer_frequency_analysis(sequences, k=3):
    def generate_kmers(seq, k):
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

    kmer_counts = {}
    total_kmers = 0

    for seq in sequences:
        seq_kmers = generate_kmers(seq, k)
        for kmer in seq_kmers:
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
            total_kmers += 1

    kmer_frequencies = {
        kmer: count / total_kmers
        for kmer, count in kmer_counts.items()
    }

    analysis_results = {
        'unique_kmers': len(kmer_frequencies),
        'total_kmers': total_kmers,
        'top_5_kmers': sorted(
            kmer_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5],
        'frequency_distribution': kmer_frequencies
    }

    return analysis_results


def kmer_spectrum_shift(real_sequences, generated_sequences, k_range):
    js_divergences = {}

    for k in k_range:
        real_kmer_freq = kmer_frequency_analysis(real_sequences, k)
        generated_kmer_freq = kmer_frequency_analysis(generated_sequences, k)

        all_kmers = list(set(
            list(real_kmer_freq['frequency_distribution'].keys()) +
            list(generated_kmer_freq['frequency_distribution'].keys())
        ))

        real_dist = [real_kmer_freq['frequency_distribution'].get(kmer, 0) for kmer in all_kmers]
        generated_dist = [generated_kmer_freq['frequency_distribution'].get(kmer, 0) for kmer in all_kmers]

        js_distance = jensenshannon(real_dist, generated_dist)

        js_divergences[k] = {
            'js_distance': js_distance,
            'real_dist': real_dist,
            'generated_dist': generated_dist
        }

    return js_divergences


def plot_kmer_spectrum_shift(js_divergences):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    colors = [
        '#D94D4D',
        '#E1812C',
        '#3BA66B',
        '#3274A1',
        '#9C7BCA',
    ]
    labels = ['DTCR_blosum', 'DTCR_random', 'DTCR_mask', 'TCR-TRANSLATE', 'GRATCR']

    for m in range(len(js_divergences)):
        k_values = list(js_divergences[m].keys())
        js_distances = [div['js_distance'] for div in js_divergences[m].values()]
        plt.scatter(k_values, js_distances, color=colors[m], label=labels[m], s=200, alpha=0.8)

    plt.title('', fontsize=22)
    plt.xlabel('', fontsize=22)
    plt.ylabel('', fontsize=22)
    plt.xticks(k_values)
    plt.legend(loc='best', fontsize=18)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./K-mer_Spectrum_Shift_scatter.svg")


def get_tcr_seqs(model):
    all_tcr_seqs = []
    file_dir = "./data/"
    dat = pd.read_csv(file_dir + model + "/results.txt")
    for i in range(len(dat)):
        epi = dat["MT_pep"][i]
        tcr = dat["CDR3"][i]
        label = dat["Label"][i]
        if label == 0:
            continue
        all_tcr_seqs.append(tcr)
    return all_tcr_seqs


models = ["DTCR_blosum", "DTCR_random", "DTCR_mask", "TCR_translate", "GRATCR"]
spectrum_shift_results = []
real_tcrs = get_tcr_seqs("true_seqs")
for i in range(len(models)):
    generated_tcrs = get_tcr_seqs(models[i])
    spectrum_shift_result = kmer_spectrum_shift(real_tcrs, generated_tcrs, [m for m in range(2, 10)])
    spectrum_shift_results.append(spectrum_shift_result)
    all_js_div = []
    for m in range(2, 10):
        all_js_div.append(spectrum_shift_result[m]["js_distance"])
    print(str(min(all_js_div[:4])) + " " + str(max(all_js_div[:4])))

plot_kmer_spectrum_shift(spectrum_shift_results)
