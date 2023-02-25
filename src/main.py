import sys
import random
import numpy as np
from apex import amp
from model import DBGB

from sklearn.model_selection import train_test_split

from transformers import AdamW
import torch
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from log import Logger

import warnings
warnings.filterwarnings("ignore")
def train(model, df, label_map):
    tokenizer = model.get_tokenizer()

    train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len)
    test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)
    trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2,shuffle=True)
    testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1,shuffle=False)
    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)#, eps=1e-8)
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    max_only_p5 = 0
    for epoch in range(0, args.epoch):
        train_loss = model.one_epoch(epoch, trainloader, optimizer, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)
        if args.valid:
            ev_result = model.one_epoch(epoch, validloader, optimizer, mode='eval')
        else:
            ev_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')

        g_p1, g_p3, g_p5, p1, p3, p5,dcg3, dcg5, psp1,psp3,psp5= ev_result
        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, dcg3: {dcg3:.4f}, dcg5: {dcg5:.4f}, psp1: {psp1:.4f},psp3: {psp3:.4f}, psp5: {psp5:.4f}, train_loss:{train_loss}'

        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        if max_only_p5 < p5:
            max_only_p5 = p5
            model.save_model(f'models/model-{get_exp_name()}.bin')


def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    name.append(str(args.maskNum))
    name.append(str(args.headtotail))
    name.append(str(args.Gscale))
    name.append(str(args.scale))
    name.append('tmp')
    return '_'.join([i for i in name if i != ''])


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)#6088
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)
parser.add_argument('--eval_step', type=int, required=False, default=20000)
parser.add_argument('--hidden_dim', type=int, required=False, default=300)
parser.add_argument('--eval_model', action='store_true')
parser.add_argument('--scale', type=float, required=False, default=0.5)
parser.add_argument('--maskNum', type=float, required=False, default=0.2)
parser.add_argument('--headtotail', type=float, required=False, default=0.2)
parser.add_argument('--Gscale', type=float, required=False, default=2)

args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)

    print(get_exp_name())

    LOG = Logger('log_'+get_exp_name())
    
    print(f'load {args.dataset} dataset...')
    name_map = {'wiki31k': 'Wiki10-31K',
                'amazoncat13k': 'AmazonCat-13K',
                'eurlex4k': 'Eurlex-4K'}

    df, label_map, freq ,train_num= createDataCSV(args.dataset)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    
    model = DBGB(n_labels=len(label_map), bert=args.bert, scale = args.scale,maskNum = args.maskNum,
                         update_count=args.update_count,freq=freq, headtotail=args.headtotail, use_swa=args.swa, swa_warmup_epoch=args.swa_warmup,train_num=train_num, swa_update_step=args.swa_step,Gscale=args.Gscale,maxepoch=args.epoch)
    train(model, df, label_map)
