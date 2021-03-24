"""
    @author: Jevis_Hoo
    @Date: 2021/2/7
    @Description: 
"""
import argparse
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import *
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup, BertTokenizer, BertForMultipleChoice

from model import BertForMRC
from utils import seed_everything, AverageMeter, MyDataset

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def train(model, args, train_loader, optimizer, scaler, scheduler):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    if not args.full_fine_tuning:
        optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(args.device), attention_mask.to(
            args.device), token_type_ids.to(args.device), y.to(args.device).long()

        with autocast():  # 使用半精度训练
            output = model(input_ids, attention_mask, token_type_ids).logits

            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, y) / args.accum_iter

            scaler.scale(loss).backward()

            if ((step + 1) % args.accum_iter == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scaler.step(optimizer)
                scaler.update()
                if args.full_fine_tuning:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    scheduler.step()

        acc = (output.argmax(1) == y).sum().item() / y.size(0)
        losses.update(loss.item() * args.accum_iter, y.size(0))
        accs.update(acc, y.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def evaluate(model, args, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(args.device), attention_mask.to(
                args.device), token_type_ids.to(args.device), y.to(args.device).long()

            output = model(input_ids, attention_mask, token_type_ids).logits

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, y)
            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accuracies.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accuracies.avg)

    return losses.avg, accuracies.avg


def train_and_evaluate(model, args, train_dataloader, eval_dataloader, optimizer, num_train_steps, output_dir,
                       timestamp, fold, scheduler=None):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    best_accuracy = 0.0
    scaler = GradScaler()
    for epoch in range(args.num_train_epochs):
        logger.info("********** Epoch {}/{} **********".format(epoch, args.num_train_epochs))

        train(model, args, train_dataloader, optimizer, scaler, scheduler)

        eval_loss, eval_accuracy = evaluate(model, args, eval_dataloader)

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, "{}_{}_model_best.pt".format(fold, timestamp)))


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=args.max_seq_length,
                         return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


def main(args):
    timestamp = str(int(time.time()))
    output_dir = os.path.abspath(
        os.path.join(args.output_dir, timestamp))

    if os.path.exists(output_dir):
        if args.do_train:
            raise ValueError("Output file ({}) already exists.".format(output_dir))
    else:
        os.makedirs(output_dir)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.info("device %s n_gpu %d distributed training", args.device, args.n_gpu)

    seed_everything(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    train_df = pd.read_csv(args.data_dir + 'train.csv')

    folds = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=args.seed) \
        .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证

    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    if args.do_train:
        for fold, (trn_idx, val_idx) in enumerate(folds):
            train_data = train_df.loc[trn_idx]
            eval_data = train_df.loc[val_idx]

            train_set = MyDataset(train_data)
            eval_set = MyDataset(eval_data)

            train_dataloader = DataLoader(train_set, batch_size=args.train_batch_size, collate_fn=collate_fn,
                                          shuffle=True, num_workers=args.num_workers)
            eval_dataloader = DataLoader(eval_set, batch_size=args.eval_batch_size, collate_fn=collate_fn,
                                         shuffle=False, num_workers=args.num_workers)

            num_train_steps = args.num_train_epochs * len(train_dataloader) // args.accum_iter

            model = BertForMRC.from_pretrained(args.bert_config_file)  # 模型
            # model = BertForMultipleChoice.from_pretrained(args.bert_config_file)  # 模型

            model.to(args.device)

            # if args.n_gpu > 1:
            #     model = torch.nn.DataParallel(model)

            if args.full_fine_tuning:
                param_optimizer = list(model.named_parameters())
                bert_grouped_parameters = [(key, value) for key, value in param_optimizer if
                                           'bert' in key and 'head_weight' not in key]

                normal_grouped_parameters = [(key, value) for key, value in param_optimizer if
                                             'bert' not in key and 'head_weight' not in key]

                head_parameters = [(key, value) for key, value in param_optimizer if 'head_weight' in key]

                no_decay = ['bias', 'LayerNorm', 'layer_norm', 'dym_weight']

                optimizer_parameters = [
                    ## bert
                    # 衰减
                    {
                        'params': [value for key, value in bert_grouped_parameters if
                                   not any(nd in key for nd in no_decay)],
                        'weight_decay': args.decay_rate, 'lr': args.bert_learning_rate
                    },
                    # 不衰减
                    {
                        'params': [value for key, value in bert_grouped_parameters if
                                   any(nd in key for nd in no_decay)],
                        'weight_decay': 0.0, 'lr': args.bert_learning_rate
                    },
                    ## normal
                    # 衰减
                    {
                        'params': [value for key, value in normal_grouped_parameters if
                                   not any(nd in key for nd in no_decay)],
                        'weight_decay': args.decay_rate, 'lr': args.learning_rate
                    },
                    # 不衰减
                    {
                        'params': [value for key, value in normal_grouped_parameters if
                                   any(nd in key for nd in no_decay)],
                        'weight_decay': 0.0, 'lr': args.learning_rate
                    },
                    ## head
                    # 衰减
                    {
                        'params': [value for key, value in head_parameters if not any(nd in key for nd in no_decay)],
                        'weight_decay': args.decay_rate, 'lr': 1e-1
                    },
                    # 不衰减
                    {
                        'params': [value for key, value in head_parameters if any(nd in key for nd in no_decay)],
                        'weight_decay': 0.0, 'lr': 1e-1
                    }
                ]

                optimizer = AdamW(optimizer_parameters)  # AdamW优化器
                scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_dataloader) // args.accum_iter,
                                                            num_train_steps)
                # optimizer = BertAdam(optimizer_parameters,
                #                      warmup=args.warmup_proportion,
                #                      schedule='warmup_cosine',
                #                      t_total=num_train_steps)
                train_and_evaluate(model, args, train_dataloader, eval_dataloader, optimizer, num_train_steps,
                                   output_dir, timestamp, fold, scheduler)
            else:
                optimizer = AdamW(model.parameters(), lr=args.bert_learning_rate,
                                  weight_decay=args.decay_rate)  # AdamW优化器
                scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_dataloader) // args.accum_iter,
                                                            num_train_steps)

                train_and_evaluate(model, args, train_dataloader, eval_dataloader, optimizer, num_train_steps,
                                   output_dir, timestamp, fold, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        # default="hfl/chinese-bert-wwm-ext",
                        default="/home/hezoujie/Models/nezha_pytorch",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        # default="/home/hezoujie/Models/chinese_base_pytorch/vocab.txt",
                        default="/home/hezoujie/Models/nezha_pytorch/vocab.txt",
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default="outputs/models",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--fold_num",
                        default=5,
                        type=int,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--full_fine_tuning",
                        default=True,
                        type=bool,
                        help="using pre-trained BERT model.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--bert_learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate",
                        default=2e-4,
                        type=float,
                        help="The initial normal learning rate for Adam.")
    parser.add_argument("--decay_rate",
                        default=1e-4,
                        type=float,
                        help="The decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_workers",
                        default=8,
                        type=int,
                        help="Total number of workers.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--accum_iter",
                        default=2,
                        type=int,
                        help="gradient accumulation to batch_size*2.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help="random seed for initialization")
    parser.add_argument('--n_gpu',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_config_file)

    main(args)
