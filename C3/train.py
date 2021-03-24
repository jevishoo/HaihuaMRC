"""
    @author: Jevis_Hoo
    @Date: 2021/2/7
    @Description: 
"""
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm, trange

from nezha.optimization import BertAdam
from nezha.nezha_modeling import BertConfig
from nezha.tokenization import BertTokenizer
from model import BertForMRC
from utils import MyProcessor, n_class, convert_examples_to_features, get_dataloader
from balanced_dataparallel import BalancedDataParallel
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def train_and_evaluate(model, args, train_dataloader, eval_dataloader, optimizer, num_train_steps, output_dir):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    best_accuracy = 0.0

    for epoch in range(args.num_train_epochs):
        logger.info("********** Epoch {}/{} **********".format(epoch, args.num_train_epochs))
        model.train()

        t = trange(len(train_dataloader))
        iter_train = train_dataloader.__iter__()

        for step, _ in enumerate(t):
            batch = iter_train.__next__()
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _ = model(input_ids, input_mask, segment_ids, label_ids, n_class)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # We have accumulated enough gradients
                model.zero_grad()

            lr_group = list(set(optimizer.get_lr()))
            lr_group.sort()
            t.set_postfix(loss='{:05.3f}'.format(loss), lr='{:.4g}'.format(lr_group[0]),
                          nlr='{:.4g}'.format(lr_group[1]))

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logits_all = []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluation"):
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            label_ids = label_ids.to(args.device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, n_class)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i in range(len(logits)):
                logits_all += [logits[i]]

            tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        if args.do_train:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      }
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy}

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        if eval_accuracy >= best_accuracy:
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # torch.save(model.state_dict(), os.path.join(output_file, "{:.4f}_model_best.pt".format(eval_accuracy)))
            torch.save(model_to_save, os.path.join(output_dir, "{:.4f}_model_best.pt".format(eval_accuracy)))
            best_accuracy = eval_accuracy


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default="./data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default="/home/hezoujie/Models/nezha_pytorch",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default="haihua",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default="/home/hezoujie/Models/nezha_pytorch/vocab.txt",
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default="outputs/models",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
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
                        default=512,
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
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--normal_learning_rate",
                        default=2e-4,
                        type=float,
                        help="The initial normal learning rate for Adam.")
    parser.add_argument("--decay_rate",
                        default=0.1,
                        type=float,
                        help="The decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
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
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    timestamp = str(int(time.time()))
    output_dir = os.path.abspath(
        os.path.join(args.output_dir, timestamp))

    if os.path.exists(output_dir):
        if args.do_train:
            raise ValueError("Output file ({}) already exists.".format(output_dir))
    else:
        os.makedirs(output_dir)

    processors = {
        "haihua": MyProcessor,
    }

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.info("device %s n_gpu %d distributed training", args.device, args.n_gpu)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name](args.data_dir)
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_config_file)
    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    for item in vars(args).items():
        logger.info('%s : %s', item[0], str(item[1]))

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # model = BertForSequenceClassification(bert_config, 1 if n_class > 1 else len(label_list))
    model = BertForMRC(bert_config, n_class)

    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # gpu0_bsz = 0
        # model = BalancedDataParallel(gpu0_bsz, model, dim=0).cuda()

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
                'params': [value for key, value in bert_grouped_parameters if not any(nd in key for nd in no_decay)],
                'weight_decay': args.decay_rate, 'lr': args.learning_rate
            },
            # 不衰减
            {
                'params': [value for key, value in bert_grouped_parameters if any(nd in key for nd in no_decay)],
                'weight_decay': 0.0, 'lr': args.learning_rate
            },
            ## normal
            # 衰减
            {
                'params': [value for key, value in normal_grouped_parameters if not any(nd in key for nd in no_decay)],
                'weight_decay': args.decay_rate, 'lr': args.normal_learning_rate
            },
            # 不衰减
            {
                'params': [value for key, value in normal_grouped_parameters if any(nd in key for nd in no_decay)],
                'weight_decay': 0.0, 'lr': args.normal_learning_rate
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

        optimizer = BertAdam(optimizer_parameters,
                             warmup=args.warmup_proportion,
                             schedule='warmup_cosine',
                             t_total=num_train_steps)
    else:
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if n not in no_decay],
             'weight_decay_rate': args.decay_rate},
            {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_parameters,
                             warmup=args.warmup_proportion,
                             schedule='warmup_cosine',
                             t_total=num_train_steps)

    # loading eval data
    eval_examples = processor.get_dev_examples()
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    eval_dataloader = get_dataloader(eval_features, batch_size=args.eval_batch_size)
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        train_dataloader = get_dataloader(train_features, batch_size=args.train_batch_size, is_train=True)

        train_and_evaluate(model, args, train_dataloader, eval_dataloader, optimizer, num_train_steps, output_dir)


if __name__ == "__main__":
    main()
