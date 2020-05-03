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
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def get_tokenization(raw_data_path, tokenized_data_path, full_tokenizer):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    head, tail = os.path.split(raw_data_path)
    single_ids = None
    if os.path.isfile(tokenized_data_path + tail):
        with open(tokenized_data_path + tail, 'r', encoding='utf8') as f:
            line = f.read().strip()
        tokens = line.split(' ')
        single_ids = [int(token) for token in tokens]
    else:
        try:
            with open(raw_data_path, 'r', encoding='utf8') as f:
                lines = f.read()
                single = lines.replace('\n', ' [SEP] ')  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        except:
            try:
                with open(raw_data_path, 'r', encoding='GB18030') as f:
                    lines = f.read()
                    single = lines.replace('\n', ' [SEP] ')  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束

            except:
                single = None

        if single is not None:
            print(tail)
            len_single = len(single)
            num_pieces = int(len_single / 100)
            single_ids = []
            for i in tqdm(range(num_pieces)):
                single_ids += full_tokenizer.convert_tokens_to_ids(
                    full_tokenizer.tokenize(single[len_single // num_pieces * i: len_single // num_pieces * (i + 1)]))
            with open(tokenized_data_path + tail, 'w', encoding='utf8') as f:
                write_content=''
                for id in single_ids[:-1]:
                    write_content+=str(id) + ' '
                write_content+=(str(single_ids[-1])+'\n')
                f.write(write_content)

    return single_ids




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert


    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path



    # if raw:
    #     print('building files')
    #     build_files(raw_data_path=raw_data_path, tokenized_data_path=tokenized_data_path, full_tokenizer=full_tokenizer,
    #                 num_pieces=num_pieces)
    #     print('files built')
    raw_data_files = [join(raw_data_path, f) for f in listdir(raw_data_path) if isfile(join(raw_data_path, f))]
    random.shuffle(raw_data_files)
    each_size=len(raw_data_files)//8
    split_raw_data_files=[]
    for i in range(8):
        split_raw_data_files.append(raw_data_files[i*each_size:(i+1)*each_size])

    def tokenization(index,raw_data_files):
        for file_path in raw_data_files[index]:
            get_tokenization(file_path,tokenized_data_path,full_tokenizer)



    xmp.spawn(tokenization, args=(split_raw_data_files,), nprocs=8, start_method='fork')

if __name__ == '__main__':
    main()
