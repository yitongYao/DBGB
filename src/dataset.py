import os
import torch
import pickle
import pandas as pd
import numpy as np
from torch import nn
import json
import tqdm
import joblib
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import transformers
transformers.logging.set_verbosity_error()
import warnings
warnings.filterwarnings('ignore')
def createDataCSV(dataset):
    labels = []
    texts = []
    dataType = []
    label_map = {}
    orTrain_label = []
    orText_label = []
    train_labels=[]
    test_labels=[]
    name_map = {'wiki31k': 'Wiki10-31K',
                'amazoncat13k': 'AmazonCat-13K',
                'eurlex4k': 'Eurlex-4K'}

    assert dataset in name_map
    dataset = name_map[dataset]

    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    if dataset=='AmazonCat-13K' or dataset=='Eurlex-4K':
        with open(f'/root/autodl-nas/data/{dataset}/train{fext}') as f:
            for i in tqdm.tqdm(f):
                texts.append(i.replace('\n', ''))
                dataType.append('train')

        with open(f'/root/autodl-nas/data/{dataset}/test{fext}') as f:
            for i in tqdm.tqdm(f):
                texts.append(i.replace('\n', ''))
                dataType.append('test')

        with open(f'/root/autodl-nas/data/{dataset}/train_labels.txt') as f:
            for i in tqdm.tqdm(f):
                labels.append(i.replace('\n', ''))
                trainLabel = i.replace('\n', '').split(',')
                for lab in trainLabel:
                    train_labels.append(lab)
                
        with open(f'/root/autodl-nas/data/{dataset}/test_labels.txt') as f:
            for i in tqdm.tqdm(f):
                labels.append(i.replace('\n', ''))
                testLabel = i.replace('\n', '').split(',')
                for lab in testLabel:
                    test_labels.append(lab)
    else:
        with open(f'/root/autodl-nas/data/{dataset}/train.csv') as f:
            df = pd.read_csv(f)
            for (i, row) in tqdm.tqdm(enumerate(df.values)):
                texts.append(row[1].replace('\n', ''))
                dataType.append('train')
                labels.append(row[2].replace('\n', ''))
        with open(f'/root/autodl-nas/data/{dataset}/dev.csv') as f:
            df = pd.read_csv(f)
            for (i, row) in tqdm.tqdm(enumerate(df.values)):
                texts.append(row[1].replace('\n', ''))
                dataType.append('test')
                labels.append(row[2].replace('\n', ''))
    sparse_labels = [i.replace('\n', '').split(',') for i in open(f'/root/autodl-nas/data/{dataset}/train_labels.txt')]
    sparse_labels = np.array(sparse_labels)
    train_num = sparse_labels.shape[0]
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_y = mlb.fit_transform(sparse_labels)
    sum_y = sparse_y.sum(axis=0).getA1()
    idx=np.argsort(-sum_y)
    sum_y = -np.sort(-sum_y)
    classes=mlb.classes_[idx].tolist()
    for i, k in enumerate(classes):
        label_map[k] = i
    assert len(texts) == len(labels) == len(dataType)
    df_row = {'text': texts, 'label': labels, 'dataType': dataType}
    df = pd.DataFrame(df_row)
    return df, label_map, sum_y,train_num

class MDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df, self.n_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        self.len = len(self.df)
        self.tokenizer, self.max_length, self.group_y = tokenizer, max_length, group_y
        self.multi_group = False
        self.token_type_ids = token_type_ids
        self.candidates_num = candidates_num

    def __getitem__(self, idx):
        max_len = self.max_length
        review = self.df.text.values[idx].lower()
        labels = [self.label_map[i] for i in self.df.label.values[idx].split(',') if i in self.label_map]

        review = ' '.join(review.split()[:max_len])

        text = review
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=max_len
            )
        else:
            # fast 
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length-1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        label_ids = torch.zeros(self.n_labels)
        label_ids = label_ids.scatter(0, torch.tensor(labels),
                                      torch.tensor([1.0 for i in labels]))

        return input_ids, attention_mask, token_type_ids, label_ids
    
    def __len__(self):
        return self.len 
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, required=False, default='amazon670k')
# args = parser.parse_args()

if __name__ == '__main__':
    df, label_map, sum_y=createDataCSV('amazoncat13k')
    print(label_map)