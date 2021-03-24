import argparse
import logging
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from bert.modeling import BertConfig, BertForSequenceClassification
from utils import MyProcessor, tokenization, n_class, convert_examples_to_features

logger = logging.getLogger(__name__)


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
    parser.add_argument("--test_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for test.")

    args = parser.parse_args()
    processors = {
        "haihua": MyProcessor,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_name = args.task_name.lower()
    processor = processors[task_name](args.data_dir)
    test_examples = processor.get_dev_examples()

    logger.info("***** Running predictor *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.test_batch_size)

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    label_list = processor.get_labels()
    model = BertForSequenceClassification(bert_config, 1 if n_class > 1 else len(label_list))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    test_examples = processor.get_test_examples()
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)

    input_ids = []
    input_mask = []
    segment_ids = []

    for f in test_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

    model.eval()
    nb_test_steps, nb_test_examples = 0, 0
    logits_all = []
    for input_ids, input_mask, segment_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            _, logits = model(input_ids, segment_ids, input_mask, n_class)

        logits = logits.detach().cpu().numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]

        nb_test_examples += input_ids.size(0)
        nb_test_steps += 1

    output_test_file = os.path.join(args.output_dir, "logits_dev.txt")
    with open(output_test_file, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i]) - 1:
                    f.write("\n")
                else:
                    f.write(" ")


if __name__ == '__main__':
    main()
