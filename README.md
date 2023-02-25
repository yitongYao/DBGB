# DBGB
The code for "A Dual-Branch Learning Model with Gradient-Balanced Loss for Long-Tailed Multi-Label Text Classification", under-reviewed by TOIS.
##Requirements
- python==3.8.10 
- torch==1.7.1
- torchvision==0.8.1
- scikit-learn==1.0.2
- scipy==1.7.3
- sentencepiece==0.1.96
- sklearn==0.0
- tokenizers==0.10.3
- pandas==1.3.5
- numpy==1.21.2
- joblib==1.1.0
- apex==0.1
- tqdm==4.61.2
- joblib==1.1.0

## Train
```sh
###EUR-Lex
python src/main.py --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 200 --batch 16 --maskNum 0.7 --Gscale 4 --headtotail 0.2
###Wiki10-31K
python src/main_base.py --epoch 20 --dataset wiki31k --swa --swa_warmup 10 --swa_step 300 --batch 16 --maskNum 0.5 --Gscale 2 --headtotail 0.2
###AmazonCat-13K
python src/main.py --lr 1e-4 --epoch 10 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --maskNum 0.7 --Gscale 2 --headtotail 0.2
```

##Test
```sh
###EUR-Lex
python src/f1ensemble.py --dataset eurlex4k --maskNum 0.7 --Gscale 4 --headtotail 0.2
###Wiki10-31K
python src/f1ensemble.py ---dataset wiki31k --maskNum 0.5 --Gscale 2 --headtotail 0.2
###AmazonCat-13K
python src/f1ensemble.py --dataset amazoncat13k --batch 16 --maskNum 0.7 --Gscale 2 --headtotail 0.2
```