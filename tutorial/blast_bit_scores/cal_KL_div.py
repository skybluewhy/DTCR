import numpy as np
from scipy.stats import entropy
import matplotlib
import pandas as pd
from scipy.stats import gaussian_kde


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


def calculate_kl_divergence(a, b, bins=20):
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)

    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10

    hist_a = hist_a / np.sum(hist_a)
    hist_b = hist_b / np.sum(hist_b)

    kl_div = entropy(hist_a, hist_b)

    return kl_div


def advanced_kl_divergence(a, b):
    kde_a = gaussian_kde(a)
    kde_b = gaussian_kde(b)

    x_min = min(np.min(a), np.min(b))
    x_max = max(np.max(a), np.max(b))
    x = np.linspace(x_min, x_max, 1000)

    pdf_a = kde_a(x)
    pdf_b = kde_b(x)

    pdf_a = pdf_a / np.sum(pdf_a)
    pdf_b = pdf_b / np.sum(pdf_b)

    kl_div = np.sum(pdf_a * np.log(pdf_a / pdf_b))

    return kl_div
