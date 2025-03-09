import functools
import os
import argparse
import torch
from transformers import BertConfig
from models.modeling_bert import BertForMaskedLM
import diffusion_condition
from torch.optim import AdamW
import fastNLP
from tqdm import tqdm
from sample import Categorical
from torch.nn.utils.rnn import pad_sequence
from data_loader_cond_epitope_design import mydataset
from models.PISTE import Transformer
from models.regression_head import TransformerModel


# python main_conditional_epitope_design.py --schedule_name blosum --schedule exp --batch_size 128 --epochs 20
# python main_conditional_epitope_design.py --schedule_name random --schedule mutual --batch_size 128 --epochs 20
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int, required=False)
    parser.add_argument("--lr", default=1e-4, type=float, required=False)
    parser.add_argument("--batch_size", default=4, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=128, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=4, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda', type=str, required=False)
    parser.add_argument("--logging_steps", default=200, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--schedule", default='exp', type=str, required=False)
    parser.add_argument("--schedule_name", default='blosum', type=str, required=False)
    parser.add_argument("--use_pred", default="1", type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_path = "checkpoint"
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig
    sample_cls = Categorical()

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

    # use_pred set to 1 will use TCR-epitope prediction tool as the base model of denoising network
    if args.use_pred == "1":
        # TCR-epitope prediction model
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
    else:  # use_pred set to 0 will use Bert model to predict MASK
        cfg = cfg_cls.from_pretrained("./bert_model")
        cfg.overall_timestep = diffusion_instance.num_steps
        model = model_cls(cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_dataset = mydataset("./data/train_val.csv")

    logger = fastNLP.logger

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
        if args.use_pred == "1":
            _, _, seq_output, _ = pred_model(new_input_ids[:, :11],
                                                  new_input_ids[:, 11:45],
                                                  new_input_ids[:, 45:])
            res = model(seq_output.detach())
            output = torch.nn.functional.one_hot(new_input_ids, 22).float()
            output[:, 45:] = res
            res = output
        else:
            res, _, _ = model(input_ids=new_input_ids, timestep=timestep, attention_mask=attention_mask)
            res = res['logits']
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
                logger.warning(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            loss = loss / args.accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

            if i % args.logging_steps == args.logging_steps - 1:
                logger.info(f'Loss at step {i} is {train_loss / args.logging_steps}')
                cur_loss = train_loss / args.logging_steps
                if cur_loss < min_loss and epoch >= 1:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, f'./{save_path}/model_design' + str(epoch) + '.th')
                    min_loss = cur_loss
                train_loss = .0
