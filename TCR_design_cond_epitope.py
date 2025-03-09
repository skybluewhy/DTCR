import torch
import os
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from sample import Categorical
from tqdm import tqdm
import argparse
from models.modeling_bert import BertForMaskedLM
import diffusion_condition as diffusion
import functools
from torch.nn.utils.rnn import pad_sequence
from data_loader_cond_epitope_design import mydataset_infer
import json
from models.PISTE import Transformer
from models.regression_head import TransformerModel


# python TCR_design_cond_epitope.py --seed_num 2000 --batch_size 1024 --model_path "./checkpoint/model_design19.th" --schedule_name 'blosum' --schedule exp
# python TCR_design_cond_epitope.py --seed_num 2000 --batch_size 1024 --model_path "./checkpoint/model_design19.th" --schedule_name 'random' --schedule mutual

with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
    token_dict = json.load(f)
token_dict = dict(zip(token_dict.values(), token_dict.keys()))
alphabets = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}

parser = argparse.ArgumentParser()
parser.add_argument("--step_size", default=1, type=int, required=False, help='Time step size during inference')
parser.add_argument("--seed_num", default=20, type=int, required=False, help='tcr seed number')
parser.add_argument("--device", default='cuda', type=str, required=False)
parser.add_argument("--batch_size", default=4, type=int, required=False, help='refine seed number for one tcr')
parser.add_argument("--model_path", default="./checkpoint/model_design9.th", type=str, required=False, help='model path')
parser.add_argument('--predict_x0', default=True, type=bool, required=False)
parser.add_argument("--schedule_name", default='blosum', type=str, required=False)
parser.add_argument("--schedule", default='exp', type=str, required=False)
parser.add_argument("--use_pred", default="1", type=str, required=False)
parser.add_argument("--num_steps", default=128, type=int, required=False)
args = parser.parse_args()

step_size = args.step_size
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
predict_x0 = args.predict_x0
num_steps = args.num_steps
schedule = args.schedule
temperature = 1.0
batch_size = args.batch_size

if not os.path.exists('./generation_results'):
    os.mkdir('generation_results')

sample_cls = Categorical()
diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
diffusion_instance = diffusion.MaskDiffusion(
    dim=22,
    schedule=diffusion_schedule,
    mask_id=21,
    sample_cls=sample_cls,
    word_freq_lambda=0.,
    device=device,
    schedule_name=args.schedule_name,
    )

if args.use_pred == "1":
    pred_model = Transformer(device=device,
                             vocab_size=22,
                             d_model=64,
                             e_layers=3,
                             d=64,
                             n_heads=9,
                             sigma=1,
                             window_threshold="default",
                             d_ff=512,
                             interact_layers=1,
                             tgt_len=11 + 34 + 30,
                             hla_max_len=34,
                             d_layers=1).to(device)
    ckpt = torch.load("./checkpoint/best_pred.pt", map_location=device)
    pred_model.load_state_dict(ckpt)
    model = TransformerModel().to(device)
else:
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig
    tok_cls = BertTokenizer
    cfg = cfg_cls.from_pretrained("./bert_model")
    cfg.overall_timestep = diffusion_instance.num_steps
    model = model_cls(cfg).to(device)

# ckpt = torch.load(args.model_path, map_location=device)
# model.load_state_dict(ckpt['model'])
model.eval()


def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
    new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
    if args.use_pred == "1":
        _, pred_labels, seq_output, _ = pred_model(new_input_ids[:, :11],
                                                   new_input_ids[:, 11:45],
                                                   new_input_ids[:, 45:])
        res = model(seq_output.detach())
        output = torch.nn.functional.one_hot(new_input_ids, 22).float()
        output[:, 45:] = res
        res = output
    else:
        res, pred_labels, seq_output = model(input_ids=new_input_ids, timestep=timestep, attention_mask=attention_mask)
        res = res['logits']
    return res, attention_mask, target_mask


def collate_fn(batch_input):
    input_ids = pad_sequence([torch.tensor(d[0]) for d in batch_input], batch_first=True)
    attention_mask = torch.stack([torch.tensor(d[1]) for d in batch_input])
    target_mask = torch.stack([torch.tensor(d[2]) for d in batch_input])
    all_sample_tensor = torch.stack([torch.tensor(d[3]) for d in batch_input])

    assert input_ids.size() == attention_mask.size() == target_mask.size()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target_mask': target_mask,
        'all_samples': all_sample_tensor,
    }


all_epi = ['LPRRSGAAGA', 'YVLDHLIVV', 'LLLDRLNQL', 'CRVLCCYVL', 'ELAGIGILTV',
           'TTDPSFLGRY', 'TPRVTGGGAM', 'SPRWYFYYL', 'GLCTLVAML', 'RAKFKQLL',
           'EAAGIGILTV', 'YLQPRTFLL', 'IVTDFSVIK', 'GILGFVFTL', 'AVFDRKSDAK',
           'LVVDFSQFSR', 'KLGGALQAK', 'LLWNGPMAV','STLPETAVVRR', 'NLVPMVATV']
for epi in all_epi:
    # test.csv is only used to select the MHC that can bind to tcrs
    test_dataset = mydataset_infer("./data/seed_tcrs.csv", "./data/test.csv", target_epitope=epi, seed_num=args.seed_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0, shuffle=False)

    with open(f'./generation_results/' + epi + ".txt", 'w+') as f_raw:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                for k, v in batch.items():
                    if v != None:
                        batch[k] = v.to(device)
                state = diffusion.generate_data(
                    input_ids=batch['input_ids'],
                    sample_input_ids=batch['all_samples'],
                    target_mask=batch['target_mask'],
                    attention_mask=batch['attention_mask'],
                    denoise_fn=functools.partial(
                        denoise_fn,
                        input_ids=batch['input_ids'],
                    ),
                    diffusion=diffusion_instance,
                    predict_x0=predict_x0,
                    step_size=step_size,
                    show_process=False,
                    temperature=temperature,
                    time_t=None,
                )
                mhc_sequences = []
                tcr_sequences = []
                target_att = batch['target_mask'].cpu().numpy().tolist()
                ori_input = batch['input_ids'].cpu().numpy().tolist()
                cnt = 0
                for pred in state:
                    sentence = pred.tolist()
                    cur_mhc_seq = ""
                    cur_tcr_seq = ""
                    cur_tcr_seq_ori = ""
                    cur_target = target_att[cnt]
                    cur_ori_input = ori_input[cnt]
                    cnt += 1
                    num = -1
                    for s in cur_ori_input:
                        num += 1
                        if s not in token_dict:
                            continue
                        if token_dict[s] not in alphabets:
                            continue
                        if num < 45:
                            continue
                        cur_tcr_seq_ori += token_dict[s]
                    num = -1
                    for s in sentence:
                        num += 1
                        if s not in token_dict:
                            continue
                        if token_dict[s] not in alphabets:
                            continue
                        if num >= 11 and num < 45:
                            cur_mhc_seq += token_dict[s]
                        elif cur_target[num] == 1:
                            cur_tcr_seq += token_dict[s]
                    mhc_sequences.append(cur_mhc_seq)
                    tcr_sequences.append(cur_tcr_seq_ori[0] + cur_tcr_seq + cur_tcr_seq_ori[-1])
                try:
                    sentences = []
                    for i in range(len(tcr_sequences)):
                        sentences.append(mhc_sequences[i] + " " + tcr_sequences[i])
                    print('\n'.join(sentences), file=f_raw, flush=True)
                except ValueError:
                    pass
