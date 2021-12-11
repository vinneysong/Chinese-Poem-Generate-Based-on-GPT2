import time
import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn

import numpy as np

from transformers import BertTokenizer, GPT2Config
from MyModel import MYGPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from RoleData import RoleDataset, create_dataloader


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, type=bool, help='是否使用GPU')
    parser.add_argument('--mode', default="CCPC", type=str, help='训练模式(唐诗:tang、宋诗：song、宋词：ci、关键词诗：CCPC')
    parser.add_argument('--epochs', default=3, type=int, help='训练epoch数量')
    parser.add_argument('--batch_size', default=1, type=int, help='训练batch数量')
    parser.add_argument('--lr', default=2e-5, type=float, help='初始学习率')
    parser.add_argument('--weight_decay', default=0.005, type=float, help='正则化系数')
    parser.add_argument('--warm_up_ratio', default=0.0, type=float, help='预热比例')
    return parser.parse_args()


args = set_args()
model_config = GPT2Config.from_json_file("./model_" + args.mode + "/config.json")
model = MYGPT2LMHeadModel(model_config)
tokenizer = BertTokenizer(vocab_file="./model_" + args.mode + "/vocab.txt", do_lower_case=False)
tokenizer.add_tokens("[SPACE]", special_tokens=True)
tokenizer.add_tokens("[KEYWORD]", special_tokens=True)
tokenizer.add_tokens("[CONTENT]", special_tokens=True)
content_id = tokenizer.convert_tokens_to_ids("[CONTENT]")
device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

model.train()
model.to(device)

max_len = model_config.n_ctx

output_dir = "./model_" + args.mode + "/"

trainset = RoleDataset(tokenizer, max_len, mode=args.mode)
train_loader = create_dataloader(trainset, args.batch_size)

# 获取模型所有参数
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
total_steps = len(train_loader) * args.epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warm_up_ratio * total_steps,
    num_training_steps=total_steps
)


def do_train(model, data_loader, optimizer, scheduler):
    model.train()
    print("start training...")
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    for epoch in range(args.epochs):
        now = datetime.now()
        losses = []
        for step, sample in enumerate(data_loader):
            input_ids = sample["input_ids"].to(device)
            if args.mode == "CCPC":
                tokens = sample["tokens"].to(device)
                outputs = model.forward2(input_ids=input_ids, token_type_ids=tokens, labels=input_ids,
                                        content_id=content_id)
            else:
                outputs = model.forward(input_ids=input_ids, labels=input_ids)

            loss, logits = outputs[:2]

            losses.append(loss.item())

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                         float(scheduler.get_last_lr()[0])))

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))
    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')


do_train(model, train_loader, optimizer, scheduler)
