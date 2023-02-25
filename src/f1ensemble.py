import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_ens import MDataset, createDataCSV

from sklearn import metrics
from model import DBGB
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--berts', type=str, required=False, default='one')
parser.add_argument('--maskNum', type=float, required=False, default=0.2)
parser.add_argument('--headtotail', type=float, required=False, default=0.2)
parser.add_argument('--Gscale', type=float, required=False, default=0.2)
parser.add_argument('--scale', type=float, required=False, default=0.5)

args = parser.parse_args()

def get_inv_propensity(n,count_vector:list, a=0.55, b=1.5,othernum=0):
    number = np.append(count_vector,[0]*(othernum-len(count_vector)))
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)

if __name__ == '__main__':
    df, label_map,n,count_vector,freq,n_labels = createDataCSV(args.dataset)
    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')
    xmc_models = []
    predicts = 0
    propensity = get_inv_propensity(n,count_vector,a=0.55,b=1.5,othernum=len(label_map))
    if args.berts == 'all':
        berts = ['bert-base','roberta' ,'xlnet']
    else:
        berts = ['bert-base']
    print(berts)
    df = df[df.dataType == 'test']
    for index in range(len(berts)):
        model_name = [args.dataset, '' if berts[index] == 'bert-base' else berts[index]]
        model_name.append(str(args.maskNum))
        model_name.append(str(args.headtotail))
        model_name.append(str(args.Gscale))
        model_name.append(str(args.scale))
        model_name = '_'.join([i for i in model_name if i != ''])
        model = DBGB(n_labels=n_labels, bert=berts[index],freq=freq)
        print(f'models/model-{model_name}.bin')
        model.load_state_dict(torch.load(f'models/model-{model_name}.bin'))
        tokenizer = model.get_tokenizer()
        test_d = MDataset(df, 'test', tokenizer, label_map, 128 if args.dataset == 'amazoncat13k' and berts[index] == 'xlnet' else 512)
        testloader = DataLoader(test_d, batch_size=16, num_workers=0,
                                shuffle=False)
        model.cuda()
        predicts=torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0])

    
    total = len(df)
    acc1 = 0
    acc3 = 0
    acc5 = 0
    psp1, psp3, psp5 = 0, 0, 0
    psp_1, psp_3, psp_5 = 0, 0, 0
    den_1, den_3, den_5 = 0, 0, 0
    headnum = int(n_labels*0.2)
    score, score3 = 0, 0
    rank = np.log2(np.arange(2, 2 + 5))
    rank3 = np.log2(np.arange(2, 2 + 3))
    p5=[]
    psp5list = []
    f1true = []
    f1pred = []
    headrecall3, tailrecall3 = 0, 0
    headrecall5, tailrecall5 = 0, 0

    for index, true_labels in enumerate(df.label.values):
        headtotal, tailtotal =0,0
        head5, tail5 = 0, 0
        head3, tail3 = 0, 0
        true_labels = set([label_map[i] for i in true_labels.split(',') ])
        logits = torch.sigmoid(predicts[index]) 
        logits = (-logits).argsort().cpu().numpy()


        for true_label in true_labels:
            if true_label<headnum:
                headtotal +=1
            elif true_label<=n_labels:
                tailtotal +=1
        label1 = set([logits[0]]) & true_labels
        label3 = set(logits[:3]) & true_labels
        label5 = set(logits[:5]) & true_labels
        acc1 += len(label1)
        acc3 += len(label3)
        acc5 += len(label5)
        p5.append(str(len(label5)/5))
        prediction = logits[:5]

        f1len = len(true_labels)
        f1prediction =  logits[:f1len]
        f1true.append(list(true_labels))
        f1pred.append(f1prediction)
        l = []
        for i in range(5):
            if prediction[i] in true_labels:
                l.append(1)
            else:
                l.append(0)
        dcg = (l / rank).sum()
        label_count = len(true_labels)
        norm = 1 / np.log2(np.arange(2, 2 + np.min((5, label_count))))
        norm = norm.sum()
        score += dcg / norm
        prediction3 = logits[:3] 
        l3 = []
        for i in range(3):
            if prediction3[i] in true_labels:
                l3.append(1)
            else:
                l3.append(0)
        dcg3 = (l3 / rank3).sum()
        norm3 = 1 / np.log2(np.arange(2, 2 + np.min((3, label_count))))
        norm3 = norm3.sum()
        score3 += dcg3 / norm3
        for label in label3:
            if label<headnum:
                head3 +=1
            elif label<=n_labels:
                tail3 +=1
        for label in label5:
            if label<headnum:
                head5 +=1
            elif label<=n_labels:
                tail5 +=1
        if headtotal!=0:
            headrecall3 += head3/headtotal
            headrecall5 += head5/headtotal
        if tailtotal!=0:
            tailrecall3 += tail3/tailtotal
            tailrecall5 += tail5/tailtotal
        den_1 += np.sum(np.sort(propensity[list(true_labels)])[-1:])
        den_3 += np.sum(np.sort(propensity[list(true_labels)])[-3:])
        den_5 += np.sum(np.sort(propensity[list(true_labels)])[-5:])
        psp_1 += np.sum(propensity[list(label1)])
        psp_3 += np.sum(propensity[list(label3)])
        psp_5 += np.sum(propensity[list(label5)])
    
    print("ndcg@3%.5f"%(score3 * 100 / total))
    print("ndcg@5%.5f"%(score * 100 / total))
    p1 = acc1/ total
    p3 = acc3/ total / 3
    p5 = acc5/ total / 5
    print("P: @1 @3 @5 %.5f %.5f %.5f"%(p1,p3,p5))
    print("headrecall3: %.5f tailrecall3: %.5f"%(headrecall3/total,tailrecall3/total))
    print("headrecall5: %.5f tailrecall5: %.5f"%(headrecall5/total,tailrecall5/total))
    psp1 = psp_1/den_1
    psp3 = psp_3/den_3#/3
    psp5 = psp_5/den_5#/5
    print("PSP: @1 @3 @5 %.5f %.5f %.5f"%(psp1,psp3,psp5))
    classes = [i+1  for i in range(n_labels)]
    mlb = MultiLabelBinarizer(classes = classes)
    f1true = mlb.fit_transform(f1true)
    f1pred = mlb.fit_transform(f1pred)
    f1macro = metrics.precision_score(f1true, f1pred, average="macro")
    f1micro = metrics.precision_score(f1true, f1pred, average="micro")
    print("F1 Score: macro micro %.5f %.5f"%(f1macro,f1micro))