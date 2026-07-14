import csv
import numpy as np
import pandas as pd
import random
import collections
from collections import defaultdict
from sklearn.utils import shuffle

import argparse
import sys


def get_all_pos(filename, epi):
    all_pos = []
    f = open(filename, 'r')
    for line in f:
        line_vec = line.replace("\n", "").split(" ")
        mhc = line_vec[0]
        tcr = line_vec[1]
        all_pos.append(epi + "," + mhc + "," + tcr + ",1")
    return all_pos


def proc_data():
    all_epi = ['LPRRSGAAGA', 'YVLDHLIVV', 'LLLDRLNQL', 'CRVLCCYVL', 'ELAGIGILTV', 'TTDPSFLGRY', 'TPRVTGGGAM',
               'SPRWYFYYL',
               'GLCTLVAML', 'RAKFKQLL', 'EAAGIGILTV', 'YLQPRTFLL', 'IVTDFSVIK', 'GILGFVFTL', 'AVFDRKSDAK', 'LVVDFSQFSR',
               'KLGGALQAK', 'LLWNGPMAV', 'STLPETAVVRR', 'NLVPMVATV']
    all_txt = "MT_pep,HLA_sequence,CDR3,Label\n"
    all_epi_pos = []
    for e in all_epi:
        print(e)
        pos_list = get_all_pos("../../generation_results/" + e + ".txt", e)
        all_epi_pos += pos_list
    f = open("eval_data_unipep_strategy.txt", 'w')
    f.write(all_txt + "\n".join(all_epi_pos))
    f.close()


def unified_pep(pos_file, neg_file, num):
    pos_data = pd.read_csv(pos_file, header=0)
    peps = pos_data['MT_pep'].values
    ps = set(peps)
    tcrs = pos_data['CDR3'].values
    ts = set(tcrs)
    hlas = pos_data['HLA_sequence'].values
    hs = set(hlas)

    p2t_pos = defaultdict(set)
    for i in range(len(tcrs)):
        p2t_pos[peps[i]].add(tcrs[i])

    p2t_neg = dict()
    for i in p2t_pos.keys():
        p2t_neg[i] = list(ts - p2t_pos[i])

    p2h_pos = defaultdict(set)
    for i in range(len(hlas)):
        p2h_pos[peps[i]].add(hlas[i])

    p2h_neg = dict()
    for i in p2h_pos.keys():
        p2h_neg[i] = list(hs - p2h_pos[i])

    neg_tcrs, neg_peps, neg_hlas = sample_neg_tcr(p2t_neg, peps, hlas, p2h_neg, num)
    neg_data = {'CDR3': neg_tcrs,
                'MT_pep': neg_peps,
                'HLA_sequence': neg_hlas
                }
    neg_df = pd.DataFrame(neg_data)
    neg_df['Label'] = 0
    neg_data2 = neg_df.drop_duplicates(subset=['CDR3', 'MT_pep', 'HLA_sequence', 'Label']).reset_index(drop=True)
    neg_data2.to_csv(neg_file, index=None)


def sample_neg_tcr(p2t_neg, peps, hlas, p2h_neg, num):
    neg_peps = []
    neg_hlas = []
    neg_tcrs = []
    # n = int(num / 2)
    i = 0
    for e in peps:
        result = random.choice([0, 1])
        if result == 0:
            continue
        neg_peps.extend([e])
        neg_t = random.choice(p2t_neg[e])
        neg_tcrs.extend([neg_t])
        neg_hla = random.choice(p2h_neg[e])
        neg_hlas.extend([neg_hla])

    for e in peps:
        result = random.choice([0, 1])
        if result == 0:
            continue
        hla = hlas[i]
        neg_peps.extend([e])
        neg_t = random.choice(p2t_neg[e])
        neg_tcrs.extend([neg_t])
        neg_hlas.extend([hla])
        i += 1

    return neg_tcrs, neg_peps, neg_hlas


def preprocess(pos_file, neg_file):
    # hla_sequence = pd.read_csv(r'./tcr-pMHC/common_hla_sequence.csv')
    pos_data = pd.read_csv(pos_file, header=0)
    neg_data = pd.read_csv(neg_file, header=0)

    final_pos_neg = pd.concat([pos_data, neg_data], axis=0).reset_index(drop=True)
    # final_pos_neg = final_pos_neg.drop_duplicates(subset=['CDR3', 'MT_pep', 'HLA_sequence']).reset_index(drop=True)

    # data = pd.merge(final_pos_neg, hla_sequence, on='HLA_type')
    data2 = final_pos_neg
    data2['peplen'] = data2['MT_pep'].str.len()
    data3 = data2[data2.peplen <= 11].reset_index(drop=True)
    data3 = data3[data3.peplen >= 8].reset_index(drop=True)
    data3 = data3.dropna(axis=0).reset_index(drop=True)

    data3["contain_X"] = ['X' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_X == True].reset_index(drop=True)
    del data3['contain_X']
    data3["contain_U"] = ['U' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_U == True].reset_index(drop=True)
    del data3['contain_U']
    data3["contain_O"] = ['O' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_O == True].reset_index(drop=True)
    del data3['contain_O']
    data3["contain_B"] = ['B' not in data3.CDR3[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_B == True].reset_index(drop=True)
    del data3['contain_B']

    data3["contain_X"] = ['X' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_X == True].reset_index(drop=True)
    del data3['contain_X']
    data3["contain_U"] = ['U' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_U == True].reset_index(drop=True)
    del data3['contain_U']
    data3["contain_O"] = ['O' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_O == True].reset_index(drop=True)
    del data3['contain_O']
    data3["contain_B"] = ['B' not in data3.MT_pep[i] for i in range(len(data3.MT_pep))]
    data3 = data3[data3.contain_B == True].reset_index(drop=True)
    del data3['contain_B']
    final_data = data3
    return final_data


def drop_duplicate(df1: pd.DataFrame, df2: pd.DataFrame, name_df_out: str) -> pd.DataFrame:
    df1["mix"] = df1["MT_pep"] + ' ' + df1["HLA_type"]
    df2["mix"] = df2["MT_pep"] + ' ' + df2["HLA_type"]
    d_dup = df2["mix"].isin(df1["mix"])
    df2_drop = df2[~d_dup].drop("mix", axis=1)
    df2_drop.to_csv(name_df_out, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='model evalution')
    parser.add_argument('--pos', type=str, default="./eval_data_unipep_strategy.txt", help='the path to the positive data file (*.csv).')
    parser.add_argument('--neg', type=str, default="./train_neg.csv", help='the path to the negative data file (*.csv).')
    parser.add_argument('--sampling', type=str, default="unipep", help='the negative sampling method.')
    parser.add_argument('--file_type', type=str, default="training", help='training or test.')
    parser.add_argument('--neg_ratio', default=1, type=int, help='Ratio of negative sampling methods.')
    args = parser.parse_args()

    if not args.pos:
        sys.exit(0)
    if not args.neg:
        sys.exit(0)
    if not args.sampling:
        sys.exit(0)
    if not args.file_type:
        sys.exit(0)

    pos_file = args.pos
    neg_file = args.neg
    sampling = args.sampling
    file_type = args.file_type
    num = args.neg_ratio

    if sampling == 'unipep':
        proc_data()
        unified_pep(pos_file, neg_file, num)
        data = preprocess(pos_file, neg_file)
        if file_type == 'training':
            data = shuffle(data)
            data.to_csv(r'./eval_data_unipep_strategy.txt', index=None)
