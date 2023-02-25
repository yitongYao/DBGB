import tqdm
import time
import cProfile
import numpy as np
from apex import amp
import torch.nn.functional as F 
from dbloss import dbloss_mask
import torch
from torch import nn
import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizerFast
import warnings
warnings.filterwarnings('ignore')
def get_bert(bert_name):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert
def get_inv_propensity(n,count_vector:list, a=0.55, b=1.5,othernum=0):
    number = np.append(count_vector,[0]*(othernum-len(count_vector)))
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)
class DBGB(nn.Module):
    def __init__(self, n_labels, group_y=None, bert='bert-base', feature_layers=5, dropout=0.5, update_count=1,
                 candidates_topk=10, freq=None, maskNum = 0,scale = 0.5,headtotail=0.2,
                 use_swa=True, swa_warmup_epoch=10, swa_update_step=200, hidden_dim=300,Gscale=4,maxepoch = 30,train_num=15449):
        super(DBGB, self).__init__()

        self.use_swa = use_swa
        self.swa_warmup_epoch = swa_warmup_epoch
        self.swa_update_step = swa_update_step
        self.swa_state = {}

        self.update_count = update_count
        self.freq = freq
        self.candidates_topk = candidates_topk
        print('swa', self.use_swa, self.swa_warmup_epoch, self.swa_update_step, self.swa_state)
        print('update_count', self.update_count)

        self.bert_name, self.bert = bert, get_bert(bert)
        self.feature_layers, self.drop_out = feature_layers, nn.Dropout(dropout)
        self.group_y = group_y
        self.headNum = int(n_labels*headtotail)
        self.scale = scale
        self.maskNum = maskNum
        self.propensity = get_inv_propensity(len(freq),freq)
        self.Gscale = Gscale
        self.maxepoch = maxepoch
        self.train_num = train_num
        kernel_sizes = [1,2,3,4,5]
        kernelNum = 768
        self.class1 = nn.Linear(len(kernel_sizes)*kernelNum, n_labels)
        self.class2 = nn.Linear(len(kernel_sizes)*kernelNum, n_labels)
        self.convs = nn.ModuleList([nn.Conv2d(1,kernelNum,(size, self.bert.config.hidden_size)) for size in kernel_sizes])
        self.WbTen = nn.Parameter(torch.full([n_labels], 1.0))


    def forward(self, input_ids, attention_mask, token_type_ids,
                labels=None, group_labels=None, candidates=None, epoch = None):
        is_training = labels is not None

        outs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[-1]
        feature_out = outs[-1].unsqueeze(1)
        pool_out = outs[-1][:,0]
        feature_out = [F.relu(conv(feature_out)).squeeze(3) for conv in self.convs]
        feature_pool = [F.max_pool1d(i,i.size(2)).squeeze(2)+pool_out for i in feature_out]
        out = torch.cat(feature_pool, 1)
        out = self.drop_out(out)
        headNum = self.headNum
        out1 = self.class1(out)#[16,6187]
        out2 = self.class2(out)
        group_logits = (out1 + self.WbTen*out2 )/2
        logits = group_logits
        if is_training:
            loss1 = dbloss_mask(out1, labels,self.freq,0,headNum,wl = self.wl,train_num=self.train_num)
            loss2 = dbloss_mask(out2, labels,self.freq,self.maskNum,headNum,wl = self.wl,train_num=self.train_num)
            loss_all = dbloss_mask(logits,labels,self.freq,0,headNum,wl = self.wl,train_num=self.train_num)
            loss = loss_all + self.scale*loss1 + (1-self.scale)*loss2 
            return logits, loss
        else:
            return logits

    def save_model(self, path):
        self.swa_swap_params()
        torch.save(self.state_dict(), path)
        self.swa_swap_params()

    def swa_init(self):
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swa_swap_params(self):
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].cuda()

    def get_fast_tokenizer(self):
        if 'roberta' in self.bert_name:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased') 
        else:
            tokenizer = BertWordPieceTokenizer(
                "data/.bert-base-uncased-vocab.txt",
                lowercase=True)
        return tokenizer

    def get_tokenizer(self):
        if 'roberta' in self.bert_name:
            print('load roberta-base tokenizer')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            print('load xlnet-base-cased tokenizer')
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        else:
            print('load bert-base-uncased tokenizer')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        return tokenizer

    def get_accuracy(self, candidates, logits, labels):
        if candidates is not None:
            candidates = candidates.detach().cpu()
        scores, indices = torch.topk(logits.detach().cpu(), k=10)
#         headacc = torch.sum(indices<self.headNum)
        acc1, acc3, acc5, total = 0, 0, 0, 0
        score = 0
        rank = np.log2(np.arange(2, 2 + 5))
        score3 =0
        rank3 = np.log2(np.arange(2, 2 + 3))
        psp_1, psp_3, psp_5 = 0, 0, 0
        den_1, den_3, den_5 = 0, 0, 0
        for i, l in enumerate(labels):
            l = set(np.nonzero(l)[0])
            if candidates is not None:
                labels = candidates[i][indices[i]].numpy()
            else:
                labels = indices[i, :5].numpy()
            
            labels1 = set([labels[0]]) & l
            acc1 += len(labels1)
            labels3 = set(labels[:3]) & l
            acc3 += len(labels3)
            labels5 = set(labels[:5]) & l
            acc5 += len(labels5)
            ld = []
            for i in range(5):
                if labels[i] in l:
                    ld.append(1)
                else:
                    ld.append(0)
            dcg = (ld / rank).sum()
            label_count = len(l)
            norm = 1 / np.log2(np.arange(2, 2 + np.min((5, label_count))))
            norm = norm.sum()
            score += dcg / norm
            prediction3 = labels[:3] 
            l3 = []
            for i in range(3):
                if prediction3[i] in l:
                    l3.append(1)
                else:
                    l3.append(0)
            dcg3 = (l3 / rank3).sum()
            norm3 = 1 / np.log2(np.arange(2, 2 + np.min((3, label_count))))
            norm3 = norm3.sum()
            score3 += dcg3 / norm3
            
            den_1 += np.sum(np.sort(self.propensity[list(l)])[-1:])
            den_3 += np.sum(np.sort(self.propensity[list(l)])[-3:])
            den_5 += np.sum(np.sort(self.propensity[list(l)])[-5:])
            psp_1 += np.sum(self.propensity[list(labels1)])
            psp_3 += np.sum(self.propensity[list(labels3)])
            psp_5 += np.sum(self.propensity[list(labels5)])
        
            total += 1

        return total, acc1, acc3, acc5, score3, score,den_1,den_3,den_5,psp_1,psp_3,psp_5

    def one_epoch(self, epoch, dataloader, optimizer,
                  mode='train', eval_loader=None, eval_step=20000, log=None):

        bar = tqdm.tqdm(total=len(dataloader))
        p1, p3, p5 = 0, 0, 0
        g_p1, g_p3, g_p5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        g_acc1, g_acc3, g_acc5 = 0, 0, 0
        train_loss = 0
        psp1, psp3, psp5 = 0, 0, 0
        psp_1, psp_3, psp_5 = 0, 0, 0
        den_1, den_3, den_5 = 0, 0, 0
        score5 = 0
        score3 =0
        if mode == 'train':
            self.train()
        else:
            self.eval()

        if self.use_swa and epoch == self.swa_warmup_epoch and mode == 'train':
            self.swa_init()

        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        pred_scores, pred_labels = [], []
        bar.set_description(f'{mode}-{epoch}')
        
        with torch.set_grad_enabled(mode == 'train'):
            self.wl = self.Gscale*epoch/(self.maxepoch-1)+1
            for step, data in enumerate(dataloader):
                batch = tuple(t for t in data)
                have_group = len(batch) > 4
                inputs = {'input_ids':      batch[0].cuda(),
                          'attention_mask': batch[1].cuda(),
                          'token_type_ids': batch[2].cuda(),'epoch': epoch}
                if mode == 'train':
                    inputs['labels'] = batch[3].cuda()
                    #print("input labels:{}".format(inputs['labels'].shape))
                    if self.group_y is not None:
                        inputs['group_labels'] = batch[4].cuda()
                        inputs['candidates'] = batch[5].cuda()

                outputs = self(**inputs)

                bar.update(1)

                if mode == 'train':
                    loss = outputs[1]
                    loss /= self.update_count
                    train_loss += loss.item()

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
#                     loss.backward()
                    if step % self.update_count == 0:
                        optimizer.step()
                        self.zero_grad()

                    if step % eval_step == 0 and eval_loader is not None and step != 0:
                        results = self.one_epoch(epoch, eval_loader, optimizer, mode='eval')
                        p1, p3, p5 = results[3:6]
                        g_p1, g_p3, g_p5 = results[:3]
                        if self.group_y is not None:
                            log.log(f'{epoch:>2} {step:>6}: {p1:.4f}, {p3:.4f}, {p5:.4f}'
                                    f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}')
                        else:
                            log.log(f'{epoch:>2} {step:>6}: {p1:.4f}, {p3:.4f}, {p5:.4f}')
                        # NOTE: we don't reset model to train mode and keep model in eval mode
                        # which means all dropout will be remove after `eval_step` in every epoch
                        # this tricks makes LightXML converge fast
                        # self.train()

                    if self.use_swa and step % self.swa_update_step == 0:
                        self.swa_step()

                    bar.set_postfix(loss=loss.item())
                    

                elif self.group_y is None:
                    logits = outputs
                    if mode == 'eval':
                        labels = batch[3]
                        _total, _acc1, _acc3, _acc5, _score3, _score,_den_1,_den_3,_den_5,_psp_1,_psp_3,_psp_5 =  self.get_accuracy(None, logits, labels.cpu().numpy())
                        total += _total; acc1 += _acc1; acc3 += _acc3; acc5 += _acc5
                        score3 += _score3
                        score5 += _score
                        den_1 += _den_1
                        den_3 += _den_3
                        den_5 += _den_5
                        psp_1 += _psp_1
                        psp_3 += _psp_3
                        psp_5 += _psp_5
                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5
                        dcg3 = score3*100/total
                        dcg5 = score5*100/total
                        psp1 = psp_1/den_1
                        psp3 = psp_3/den_3#/3
                        psp5 = psp_5/den_5#/5
                        
                        bar.set_postfix(p1=p1, p3=p3, p5=p5)
                    elif mode == 'test':
                        pred_scores.append(logits.detach().cpu())


        if self.use_swa and mode == 'eval':
            self.swa_swap_params()
        bar.close()

        if mode == 'eval':
            return g_p1, g_p3, g_p5, p1, p3, p5,dcg3, dcg5, psp1,psp3,psp5
        elif mode == 'test':
            return torch.cat(pred_scores, dim=0).numpy(), torch.cat(pred_labels, dim=0).numpy() if len(pred_labels) != 0 else None
        elif mode == 'train':
            return train_loss
