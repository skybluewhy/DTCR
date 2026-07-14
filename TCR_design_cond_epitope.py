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
from data_loader_cond_epitope_design import mydataset_reinforce_learn, alphabets
import json


with open("./bert_model/vocab.json", 'r', encoding='UTF-8') as f:
    token_dict = json.load(f)
token_dict = dict(zip(token_dict.values(), token_dict.keys()))

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=1, type=int, required=False)
parser.add_argument("--step_size", default=1, type=int, required=False, help='Time step size during inference')
parser.add_argument("--task_name", default='qqp', type=str, required=False)
parser.add_argument("--ckpt_path", default='', type=str, required=False)
parser.add_argument("--MBR_size", default=200, type=int, required=False, help=r'The MBR size \mathcal{S}. Generates that many sentences for 1 source sentence.')
parser.add_argument("--max_tcr_len", default=20, type=int, required=False, help='Max seq length in generation')
parser.add_argument("--result_file", default="raw.txt", type=str, required=False, help='Result file')
parser.add_argument("--target_epitope", default="KLGGALQAK", type=str, required=False, help='target epitope')
parser.add_argument("--seed_num", default=20, type=int, required=False, help='seed tcr number')
parser.add_argument("--refine", default=False, type=bool, required=False, help='refine selected tcrs')
parser.add_argument("--batch_size", default=1024, type=int, required=False, help='refine seed number for one tcr')
parser.add_argument("--model_path", default="./checkpoint/model_design9.th", type=str, required=False, help='model path')
parser.add_argument('--predict_x0', default=True, type=bool, required=False)
parser.add_argument("--schedule_name", default='blosum', type=str, required=False)
parser.add_argument("--schedule", default='exp', type=str, required=False)
args = parser.parse_args()

step_size = args.step_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'bert-base-uncased'
predict_x0 = args.predict_x0
sample_strategy = 'Categorical'
num_steps = 128
schedule = args.schedule
topk = args.topk
task_name = args.task_name
model_ckpt_path = args.ckpt_path
temperature = 1.0
batch_size = args.batch_size
MBR_size = args.MBR_size

if not os.path.exists('./generation_results'):
    os.mkdir('generation_results')

model_cls = BertForMaskedLM
cfg_cls = BertConfig
tok_cls = BertTokenizer

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

cfg = cfg_cls.from_pretrained("./bert_model")
cfg.overall_timestep = diffusion_instance.num_steps

model = model_cls(cfg).to(device)
ckpt = torch.load(args.model_path, map_location=device)
model.load_state_dict(ckpt['model'])

ckpt = torch.load("./checkpoint/best_pred.pt", map_location=device)
model.pred_model.load_state_dict(ckpt)

model.eval()


def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
    new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
    res, _ = model(new_input_ids=new_input_ids, timestep=timestep, attention_mask=attention_mask)
    output = torch.nn.functional.one_hot(new_input_ids, 22).float()
    output[:, 45:] = res['logits']
    return output, attention_mask, target_mask, new_input_ids


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


all_epi = ['LPRRSGAAGA', 'YVLDHLIVV', 'LLLDRLNQL', 'CRVLCCYVL', 'ELAGIGILTV', 'TTDPSFLGRY', 'TPRVTGGGAM', 'SPRWYFYYL', 'GLCTLVAML', 'RAKFKQLL', 'EAAGIGILTV', 'YLQPRTFLL', 'IVTDFSVIK', 'GILGFVFTL', 'AVFDRKSDAK', 'LVVDFSQFSR', 'KLGGALQAK', 'LLWNGPMAV', 'STLPETAVVRR', 'NLVPMVATV']
for epi in all_epi:
    test_dataset = mydataset_reinforce_learn("./data/seed_tcrs.csv", "./data/test.csv",
                                             target_epitope=epi, seed_num=args.seed_num, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                              num_workers=0, shuffle=False)

    with open(f'./generation_results/' + epi + ".txt", 'w+') as f_raw:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                for k, v in batch.items():
                    if v != None:
                        batch[k] = v.to(device)
                state = diffusion.generate_data(
                    input_ids=batch['all_samples'],
                    target_mask=batch['target_mask'],
                    attention_mask=batch['attention_mask'],
                    denoise_fn=functools.partial(
                        denoise_fn,
                        input_ids=batch['input_ids'],
                    ),
                    diffusion=diffusion_instance,
                    predict_x0=predict_x0,
                    sample_cls=sample_cls,
                    step_size=step_size,
                    topk=topk,
                    show_process=False,
                    temperature=temperature,
                    time_t=None,
                    network=model.pred_model,
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
                    tcr_sequences.append(cur_tcr_seq_ori[:1] + cur_tcr_seq + cur_tcr_seq_ori[-1:])
                try:
                    sentences = []
                    for i in range(len(tcr_sequences)):
                        sentences.append(mhc_sequences[i] + " " + tcr_sequences[i])
                    print('\n'.join(sentences), file=f_raw, flush=True)
                except ValueError:
                    pass
