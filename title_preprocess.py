import warnings
warnings.simplefilter('ignore')

import gc
import re
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import BertTokenizer, BertModel

torch.set_num_threads(8)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained("bert-base-multilingual-uncased")


df_prods = pd.read_csv('data/products_train.csv')
df_sess = pd.read_csv('data/sessions_train.csv')
df_test = pd.read_csv('data/sessions_test_task1.csv')


df_prods_jp = df_prods[df_prods['locale']=='DE']

df_prods_jp.title = df_prods_jp.title.fillna("")
df_prods_jp.title.isna().sum()

jp_embs = []
cnt = 0

for idx, row in tqdm(df_prods_jp.iterrows(), total=df_prods_jp.shape[0]):
    title = row['title']    
    encoded_input = tokenizer(title, return_tensors='pt')   
    output = model(**encoded_input)
    
    jp_embs.append(output.pooler_output.squeeze().tolist())
    cnt+=1
    
torch.save(torch.tensor(jp_embs), 'de.pt')
