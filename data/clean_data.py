import pandas as pd

dat = pd.read_json('../raw/multinli_1.0_train.jsonl', lines=True)
dat['annotator_labels'] = dat['annotator_labels'].apply(''.join)
dat = dat[['annotator_labels','sentence1','sentence2']]
dat.to_csv('multinli_train.csv',encoding='utf-8',index=False)