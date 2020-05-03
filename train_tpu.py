import transformers
import torch
import os
import json
import random
import argparse
import numpy as np
from datetime import datetime
from torch.nn import DataParallel
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

from torch.utils.data import Dataset, DataLoader
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import math
from transformers import activations
import gc


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


activations.ACT2FN['gelu'] = _gelu_python
'''
如果训练材料是全部堆在一起不分篇章的话用这个文件
'''


def get_tokenization(raw_data_path, tokenized_data_path, full_tokenizer):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    head, tail = os.path.split(raw_data_path)
    single_ids = []

    if os.path.isfile(tokenized_data_path + tail):
        try:
            with open(tokenized_data_path + tail, 'r', encoding='utf8') as f:
                line = f.read().strip()
            tokens = line.split(' ')
            single_ids = []
            for token in tokens:
                if token!='':
                    single_ids.append(int(token))
        except:
            os.remove(tokenized_data_path + tail)
            pass

    return single_ids


def build_files(raw_data_path, tokenized_data_path, full_tokenizer, num_pieces):
    with open(raw_data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
    single = ''.join(lines)
    len_single = len(single)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        single_ids = full_tokenizer.convert_tokens_to_ids(
            full_tokenizer.tokenize(single[len_single // num_pieces * i: len_single // num_pieces * (i + 1)]))
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in single_ids[:-1]:
                f.write(str(id) + ' ')
            f.write(str(single_ids[-1]))
            f.write('\n')

    print('finish')


class TextDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files, tokenized_path, full_tokenizer, n_ctx):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.tokenized_path = tokenized_path
        self.tokens = []
        self.n_ctx = n_ctx
        for file_name in files:
            token = get_tokenization(file_name, tokenized_path, full_tokenizer)
            if token is not None:
                self.tokens += token

    def __len__(self):
        return len(self.tokens) // self.n_ctx

    def __getitem__(self, idx):
        labels = self.tokens[idx * self.n_ctx: (idx + 1) * self.n_ctx]

        # batch_labels = []
        # batch_inputs = []
        # for ids in batch:
        #     int_ids_for_labels = [int(x) for x in ids]
        #     int_ids_for_inputs = [int(x) for x in ids]
        #     batch_labels.append(int_ids_for_labels)
        #     batch_inputs.append(int_ids_for_inputs)
        return torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    output_dir = args.output_dir

    # if raw:
    #     print('building files')
    #     build_files(raw_data_path=raw_data_path, tokenized_data_path=tokenized_data_path, full_tokenizer=full_tokenizer,
    #                 num_pieces=num_pieces)
    #     print('files built')
    raw_data_files = [join(raw_data_path, f) for f in listdir(raw_data_path) if isfile(join(raw_data_path, f))]
    random.shuffle(raw_data_files)

    def train_model(index):
        device = xm.xla_device()
        torch.manual_seed(0)

        if not os.path.exists(tokenized_data_path):
            os.mkdir(tokenized_data_path)
        if not args.pretrained_model:
            model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
        else:
            model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
            model.load_state_dict(torch.load(output_dir + 'final_model'))
        model.train()
        model.to(device)
        multi_gpu = False
        full_len = 0
        # print('calculating total steps')
        # for i in tqdm(range(num_pieces)):
        #     with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
        #         full_len += len([int(item) for item in f.read().strip().split()])
        # total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
        # print('total steps = {}'.format(total_steps))

        optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
        # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps)
        # if fp16:
        #     try:
        #         from apex import amp
        #     except ImportError:
        #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #     model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = DataParallel(model)
        #     multi_gpu = True
        if xm.is_master_ordinal():
            print('starting training')

        doc_size = 10
        raw_data_batch_len = len(raw_data_files) // doc_size
        for epoch in range(epochs):
            if xm.is_master_ordinal():
                print('epoch {}'.format(epoch + 1))
                now = datetime.now()
                print('time: {}'.format(now))
            for batch_len in range(raw_data_batch_len):
                train_dataset = TextDataset(raw_data_files[batch_len * doc_size:(batch_len + 1) * doc_size],
                                            tokenized_data_path, full_tokenizer,
                                            n_ctx)

                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=True)

                # Creates dataloaders, which load data in batches
                # Note: test loader is not shuffled or sampled
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    num_workers=8,
                    drop_last=True)

                # tokens = get_tokenization(raw_data_file, tokenized_data_path, full_tokenizer)
                # if tokens is None:
                #     continue
                # start_point = 0
                # samples = []
                # while start_point < len(tokens) - n_ctx:
                #     samples.append(tokens[start_point: start_point + n_ctx])
                #     start_point += stride
                # if start_point < len(tokens):
                #     samples.append(tokens[len(tokens) - n_ctx:])
                # random.shuffle(samples)
                para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
                running_loss=0
                for step, batch_inputs in enumerate(para_train_loader):

                    # for step in range(len(samples) // batch_size):

                    #  prepare data
                    # batch = samples[step * batch_size: (step + 1) * batch_size]
                    # batch_labels = []
                    # batch_inputs = []
                    # for ids in batch:
                    #     int_ids_for_labels = [int(x) for x in ids]
                    #     int_ids_for_inputs = [int(x) for x in ids]
                    #     batch_labels.append(int_ids_for_labels)
                    #     batch_inputs.append(int_ids_for_inputs)
                    # print(batch_inputs)
                    batch_inputs = batch_inputs.to(device)

                    # print(batch_labels.size(), batch_inputs.size())
                    #  forward pass
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                    loss, logits = outputs[:2]

                    #  get loss
                    # if multi_gpu:
                    #     loss = loss.mean()
                    # if gradient_accumulation > 1:
                    #     loss = loss / gradient_accumulation

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    xm.optimizer_step(optimizer)

                    # if (step + 1) % gradient_accumulation == 0:
                    #     running_loss += loss.item()
                    # optimizer.step()
                    # xm.optimizer_step(optimizer)
                    # optimizer.zero_grad()
                    # scheduler.step()
                    if xm.is_master_ordinal():
                        if (step + 1) % log_step == 0:
                            print('now time: {}:{}. Step {}/{} of pice {}/{} epoch {}, loss {}'.format(
                                datetime.now().hour,
                                datetime.now().minute,
                                (step + 1),
                                len(para_train_loader),
                                batch_len + 1,
                                raw_data_batch_len,
                                epoch + 1,
                                running_loss/log_step
                            ))
                            running_loss=0
                        else:
                            running_loss += loss.item()
                xm.save(model.state_dict(), output_dir + 'final_model')

                if xm.is_master_ordinal():
                    gc.collect()

    xmp.spawn(train_model, args=(), nprocs=8, start_method='fork')


if __name__ == '__main__':
    main()
