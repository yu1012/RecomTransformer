import torch
import numpy as np
import random
import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

os.sys.path.append("model/")

from dataset import *
from bert4rec_model import *

from utils import *



dataset = 'DE'
device = 'cuda:0'


with open(file='data_prc/'+f'{dataset}'+'_data.pkl', mode='rb') as f:
    data = pickle.load(f)

train_session, val_session, test_session = data
test_session = val_session

sessions = []
for i in range(len(test_session)-1):
    if set(test_session[i]).issubset(set(test_session[i+1])):
        continue
    else:
        sessions.append(test_session[i])
test_session = sessions


num_item = 1+max([max(sess) for sess in train_session+val_session+test_session])
max_len = max([len(sess) for sess in train_session+val_session+test_session])

test_dataset = Bert4RecDataset(test_session, max_len, split_mode="test")
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)


model = RecommendationTransformer(vocab_size=num_item, heads=1, layers=1, num_pos=max_len)
model = model.to(device)

checkpoints = torch.load("checkpoints/DE_checkpoint.tar")
model.load_state_dict(checkpoints['model_state_dict'])


ind = torch.LongTensor([i for i in range(2, num_item)]).to(device)
# ind = 1

mrr = []
for batch in tqdm(test_loader):
    src = batch['source'].to(device)
    src_mask = batch['source_mask'].to(device)
    tgt = batch['target'].to(device)
    tgt_mask = batch['target_mask'].to(device)
    
    output = model(src, src_mask, ind)
    
    idx = (src==1).nonzero()[:,1]
    for i in range(src.shape[0]):
        logits = output[i,:,idx[i]]
        topk = torch.topk(logits, k=100).indices

        answer = tgt[i,idx[i]].item()
        rank = (topk==answer).nonzero()

        if len(rank) != 0:
            mrr.append(1/(1+rank.cpu().item()))
        else:
            mrr.append(0)
            
print(np.mean(mrr))