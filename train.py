import torch
import numpy as np
import random
import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
os.sys.path.append("model/")

from dataset import *
from bert4rec_model import *

from utils import *

args = parse_args()
init_for_distributed(args)

with open(file=args.data_path, mode='rb') as f:
    data = pickle.load(f)

device = 'cuda:{}'.format(args.gpu)

sessions = []
for sess, item in zip(data[0], data[1]):
    sessions.append(sess+[item])

num_item = max([max(sess) for sess in sessions])
max_len = max([len(sess) for sess in sessions])

train_session, val_session = train_test_split(sessions, test_size=0.2)
val_session, test_session = train_test_split(val_session, test_size=0.5)

train_dataset = Bert4RecDataset(train_session, split_mode="train")
val_dataset = Bert4RecDataset(val_session, split_mode="val")

train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4, sampler=train_sampler, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, sampler=train_sampler, pin_memory=True)

model = RecommendationTransformer(vocab_size=num_item, num_pos=max_len)
model = model.to(device)
model = DDP(module=model, device_ids=[args.gpu])

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

for epoch in range(args.epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    
    for _, batch in enumerate(tqdm(train_loader)):
        src = batch['source'].to(device)        # batch_size * max_len(157)
        src_mask = batch['source_mask'].to(device)
        tgt = batch['target'].to(device)
        tgt_mask = batch['target_mask'].to(device)

        optimizer.zero_grad()
        
        output = model(src, src_mask)

        loss = calculate_loss(output, tgt, tgt_mask)
        loss.backward()
        
        optimizer.step()
    break