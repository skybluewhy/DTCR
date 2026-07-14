import pandas as pd


dat = pd.read_csv("./beta.csv")
all_seqs = []
for i in range(len(dat)):
    seq = dat["cdr3b"][i]
    all_seqs.append(">Sequence" + str(i) + "\n" + seq)
f = open("./tcrdb_tcrs.fasta", 'w')
f.write("\n".join(all_seqs))
f.close()
