import pandas as pd


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
    f = open("combined_tcrs.txt", 'w')
    f.write(all_txt + "\n".join(all_epi_pos))
    f.close()


proc_data()
dat = pd.read_csv("./combined_tcrs.txt")
all_seqs = []
for i in range(len(dat)):
    seq = dat["CDR3"][i]
    epi = dat["MT_pep"][i]
    label = dat["Label"][i]
    all_seqs.append(">gen_tcr" + str(i) + "\n" + seq)

f = open("./gen_tcrs.fasta", 'w')
f.write("\n".join(all_seqs))
f.close()
