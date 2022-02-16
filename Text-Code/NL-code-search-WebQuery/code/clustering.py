from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import pickle
import random
from sklearn.cluster import KMeans
from sklearn import metrics

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaModel,
                          RobertaTokenizer)

from models import Model
from utils import acc_and_f1, TextDataset, convert_examples_to_features
import multiprocessing
cpu_cont = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


def get_logit(item):
    return item['logit']


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clustering(args, model, tokenizer):
    clustering_data_path = os.path.join(args.data_dir, args.clustering_file)
    clustering_dataset = TextDataset(tokenizer, args, clustering_data_path, type='clustering')

    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    sampler = SequentialSampler(clustering_dataset) if args.local_rank == -1 else DistributedSampler(clustering_dataset)
    dataloader = DataLoader(clustering_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    model.eval()
    all_idx = []
    all_code_vec = []
    for batch in tqdm(dataloader):
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        idxes = batch[3]
        for idx in idxes:
            all_idx.append(idx)
        with torch.no_grad():
            code_vec, _ = model(code_inputs, nl_inputs, labels, return_vec=True)
            all_code_vec.append(code_vec.cpu())
    all_code_vec = torch.cat(all_code_vec, 0).squeeze().numpy()

    max_score = 0.0
    best_k = 0
    for k in range(32, 64):
        km = KMeans(n_clusters=k)
        y_pred = km.fit_predict(all_code_vec)
        score = metrics.calinski_harabasz_score(all_code_vec, y_pred)
        print("k:{} score:{}".format(k, score))
        if score > max_score:
            max_score = score
            best_k = k

    results = []
    km = KMeans(n_clusters=best_k)
    km_model = km.fit(all_code_vec)
    for i in range(best_k):
        results.append({
            'cluster_center': km_model.cluster_centers_[i].tolist(),
            'idxes': []
        })
    for i, label in enumerate(km_model.labels_):
        results[label]['idxes'].append(all_idx[i])

    return results


def prepare_test_json(args, model, tokenizer):
    output_path = os.path.join(args.output_dir, 'cluster.json')
    with open(output_path, 'r') as f:
        results = json.load(f)

    feat = convert_examples_to_features({
        'label': 0, 'code': '',
        'doc': args.query
    }, tokenizer, args)
    label = torch.tensor(feat.label)
    _, nl_vec = model(
        torch.tensor(feat.code_ids),
        torch.tensor(feat.nl_ids),
        torch.tensor(feat.label),
        return_vec=True
    )

    best_idx = -1
    best_logit = 0.0
    for idx, result in enumerate(results):
        code_vec = torch.tensor(result['cluster_center'])
        logits, _, _ = model(code_vec, nl_vec, label, use_input=True)
        logit = logits.squeeze().numpy().tolist()[0]
        print(logit)
        if logit > best_logit:
            best_logit = logit
            best_idx = idx
    idx_map = {}
    for idx in results[best_idx]['idxes']:
        idx_map[idx] = True

    test_data_path = os.path.join(args.data_dir, args.test_file)
    output_test_file_path = os.path.join(args.data_dir, args.output_test_file)
    output_js = []
    with open(test_data_path, 'r') as f:
        data = json.load(f)
    for js in data:
        if idx_map[js['idx']]:
            js['doc'] = args.query
            output_js.append(js)
    with open(output_test_file_path, 'w') as f:
        json.dump(output_js, f)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pred_model_dir", default=None, type=str, required=True,
                        help='model for prediction')
    parser.add_argument("--clustering_file", default=None, type=str, required=True,
                        help="An input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    ## Other parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pn_weight", type=float, default=1.0,
                        help="Ratio of positive examples in the sum of bce loss")
    parser.add_argument("--encoder_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--n_cpu', type=int, default=1, help="CPU number when CUDA is unavailable")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader num_workers")

    parser.add_argument('--output_test_file', default='query_staqc_doc.json', type=str,
                        help="loc to store generated test file")
    parser.add_argument("--prediction_file", default='evaluator/staqc_query_predictions.txt', type=str,
                        help='path to save predictions result, note to specify task name')
    parser.add_argument('--do_clustering', action='store_true')
    parser.add_argument("--prepare_test_json", action='store_true',
                        help="gen test json for run_classifier")
    parser.add_argument("--output_answer", action='store_true',
                        help="get answer from prediction.txt")
    parser.add_argument('--query', default='', type=str, help="query text")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.n_gpu = args.n_cpu
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.max_seq_length <= 0:
        args.max_seq_length = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_length = min(args.max_seq_length, tokenizer.max_len_single_sentence)
    if args.encoder_name_or_path:
        model = model_class.from_pretrained(args.encoder_name_or_path,
                                            from_tf=bool('.ckpt' in args.encoder_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Clustering
    if args.do_clustering:
        model_dir = os.path.join(args.output_dir, args.pred_model_dir)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin')))
        tokenizer = tokenizer.from_pretrained(model_dir)
        model.to(args.device)
        results = clustering(args, model, tokenizer)
        output_path = os.path.join(args.output_dir, 'cluster.json')
        with open(output_path, 'w') as f:
            json.dump(results, f)

    if args.prepare_test_json:
        model_dir = os.path.join(args.output_dir, args.pred_model_dir)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin')))
        tokenizer = tokenizer.from_pretrained(model_dir)
        model.to(args.device)
        prepare_test_json(args, model, tokenizer)

    if args.output_answer:
        test_data_path = os.path.join(args.data_dir, args.test_file)
        qid_to_code_data_path = os.path.join(args.data_dir, 'qid_to_code.pickle')
        code_data = pickle.load(open(qid_to_code_data_path, 'rb'))

        idx_pid_map = {}
        with open(test_data_path, 'r') as f:
            data = json.load(f)
            for js in data:
                idx_pid_map[js['idx']] = js['pid']

        list = []
        with open(args.prediction_file, 'r') as f:
            for line in f.readlines():
                pred = line.strip().split('\t')
                idx, logit = pred[0], float(pred[1])
                list.append({
                    'pid': idx_pid_map[idx],
                    'logit': logit
                })

        list.sort(reverse=True, key=get_logit)
        for i in range(10):
            print("******" + str(list[i]['logit']) + "******")
            print(code_data[list[i]['pid']])
            print("******************************")


if __name__ == "__main__":
    main()
