import olga.generation_probability as pgen
import olga.load_model as load_model
import pandas as pd


# pip install olga
params_file_name = 'default_models/human_T_beta/model_params.txt'
marginals_file_name = 'default_models/human_T_beta/model_marginals.txt'
V_anchor_pos_file = 'default_models/human_T_beta/V_gene_CDR3_anchors.csv'
J_anchor_pos_file = 'default_models/human_T_beta/J_gene_CDR3_anchors.csv'
genomic_data = load_model.GenomicDataVDJ()
genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
generative_model = load_model.GenerativeModelVDJ()
generative_model.load_and_process_igor_model(marginals_file_name)
pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)

model_names = ["DTCR_blosum", "DTCR_random", "DTCR_mask", "TCR_translate", "GRATCR", "true_seqs"]
for model_name in model_names:
    all_res = []
    dat = pd.read_csv("./data/" + model_name + "/results.txt")
    if model_name == "GRATCR":
        cur_res = "MT_pep,CDR3,Label,prob\n"
        for i in range(len(dat)):
            epi = dat["MT_pep"][i]
            tcr = dat["CDR3"][i]
            label = dat["Label"][i]
            if label == 0:
                continue
            prob = pgen_model.compute_aa_CDR3_pgen(tcr)
            all_res.append(epi + "," + tcr + "," + str(label) + "," + str(prob))
    else:
        cur_res = "MT_pep,HLA_sequence,CDR3,Label,prob\n"
        for i in range(len(dat)):
            epi = dat["MT_pep"][i]
            hla = dat["HLA_sequence"][i]
            tcr = dat["CDR3"][i]
            label = dat["Label"][i]
            if label == 0:
                continue
            prob = pgen_model.compute_aa_CDR3_pgen(tcr)
            all_res.append(epi + "," + hla + "," + tcr + "," + str(label) + "," + str(prob))
    f = open("./data/" + model_name + "/output.txt", 'w')
    f.write(cur_res + "\n".join(all_res))
    f.close()
