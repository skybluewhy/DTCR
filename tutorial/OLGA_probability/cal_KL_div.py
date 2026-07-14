import numpy as np
from scipy.stats import entropy
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
    """计算CDF，合并相同的值"""
    # 计算唯一值及其频率
    unique_values, counts = np.unique(data, return_counts=True)

    # 计算CDF
    cdf = np.cumsum(counts) / len(data)  # 累加频数，除以总数得到CDF
    return unique_values, cdf


def calculate_kl_divergence(a, b, bins=20):
    # 使用numpy的histogram创建概率分布
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)

    # 添加极小值避免log(0)
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10

    # 归一化
    hist_a = hist_a / np.sum(hist_a)
    hist_b = hist_b / np.sum(hist_b)

    # 计算KL散度
    kl_div = entropy(hist_a, hist_b)

    return kl_div


def advanced_kl_divergence(a, b):
    # 使用核密度估计
    kde_a = gaussian_kde(a)
    kde_b = gaussian_kde(b)

    # 在共同的范围内采样
    x_min = min(np.min(a), np.min(b))
    x_max = max(np.max(a), np.max(b))
    x = np.linspace(x_min, x_max, 1000)

    # 计算概率密度
    pdf_a = kde_a(x)
    pdf_b = kde_b(x)

    # 归一化
    pdf_a = pdf_a / np.sum(pdf_a)
    pdf_b = pdf_b / np.sum(pdf_b)

    # 计算KL散度
    kl_div = np.sum(pdf_a * np.log(pdf_a / pdf_b))

    return kl_div
