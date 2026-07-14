import functools
import os
import random
import numpy as np
import argparse
import torch
from transformers import BertConfig
from models.modeling_bert import BertForMaskedLM
import diffusion_condition
from torch.optim import AdamW
from tqdm import tqdm
from sample import Categorical
from torch.nn.utils.rnn import pad_sequence
from data_loader_cond_epitope_design import mydataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int, required=False)
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False)
    parser.add_argument("--task_name", default='qqp', type=str, required=False)
    parser.add_argument("--lr", default=1e-4, type=float, required=False)
    parser.add_argument("--batch_size", default=4, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=128, type=int, required=False)
    parser.add_argument("--eval_step_size", default=16, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=4, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--hybrid_lambda_mutation", default=0, type=float, required=False)
    parser.add_argument("--eval_steps", default=500, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=200, type=int, required=False)
    parser.add_argument("--save_steps", default=2000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='exp', type=str, required=False)
    parser.add_argument("--from_scratch", default=True, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    parser.add_argument("--schedule_name", default='blosum', type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = './logs'
    save_path = "checkpoint"
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig

    sample_cls = Categorical()
    print(args.schedule_name)

    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=22,
        schedule=diffusion_schedule,
        mask_id=21,
        sample_cls=sample_cls,
        word_freq_lambda=args.word_freq_lambda,
        device=device,
        schedule_name=args.schedule_name,
    )

    cfg = cfg_cls.from_pretrained("./bert_model")
    cfg.overall_timestep = diffusion_instance.num_steps

    model = model_cls(cfg).to(device)
    ckpt = torch.load("./checkpoint/best_pred.pt", map_location=device)
    model.pred_model.load_state_dict(ckpt)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

    train_dataset = mydataset("./data/train_val.csv")

    print('# of train data: {}'.format(len(train_dataset)))

    def collate_fn(batch_input):
        input_ids = [torch.tensor(k) for d in batch_input for k in d[0]]
        attention_mask = [torch.tensor(k) for d in batch_input for k in d[1]]
        target_mask = [torch.tensor(k) for d in batch_input for k in d[2]]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        target_mask = pad_sequence(target_mask, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0, shuffle=True)
    model.train()

    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        res, _ = model(new_input_ids=new_input_ids, timestep=timestep, attention_mask=attention_mask)
        output = torch.nn.functional.one_hot(new_input_ids, 22).float()
        output[:, 45:] = res['logits']
        res = output
        return res


    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_loss = .0
    nan_count = 0
    i = -1
    for epoch in range(args.epochs):
        min_loss = 100
        for batch in tqdm(train_loader):
            i += 1
            for k, v in batch.items():
                batch[k] = v.to(device)
            t = diffusion_instance.sample_t()
            metrics = diffusion_condition.compute_kl_reverse_process(
                batch['input_ids'],
                t.to(device),
                denoise_fn=functools.partial(
                    denoise_fn,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_mask=batch['target_mask']
                ),
                diffusion=diffusion_instance,
                target_mask=batch['target_mask'],
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=torch.zeros_like(batch['input_ids']),
            )

            loss = metrics['loss']
            loss_list = [loss]
            if torch.stack(loss_list).isnan().any():
                nan_count += 1
                print(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            loss = loss / args.accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                warmup_scheduler.step()

            if i % args.logging_steps == args.logging_steps - 1:
                print(f'Loss at step {i} is {train_loss / args.logging_steps}')
                cur_loss = train_loss / args.logging_steps
                if cur_loss < min_loss and epoch >= 1:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'warmup_scheduler': warmup_scheduler.state_dict(),
                    }, f'./{save_path}/model_design' + str(epoch) + '.th')
                    min_loss = cur_loss
                train_loss = .0
