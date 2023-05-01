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

import wandb

args = parse_args()
init_for_distributed(args)


if args.rank == 0:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RecomTransformer",
        name='DE',
        # track hyperparameters and run metadata
        config={
        "dataset": args.data_path,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "lr": args.lr
        }
    )

with open(file=args.data_path, mode='rb') as f:
    data = pickle.load(f)

device = 'cuda:{}'.format(args.gpu)

sessions = []
for sess, item in zip(data[0], data[1]):
    sessions.append(sess+[item])

num_item = 1+max([max(sess) for sess in sessions])
max_len = max([len(sess) for sess in sessions])

train_session, val_session = train_test_split(sessions, test_size=0.2)
val_session, test_session = train_test_split(val_session, test_size=0.5)

train_dataset = Bert4RecDataset(train_session, max_len, split_mode="train")
val_dataset = Bert4RecDataset(val_session, max_len, split_mode="val")

train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=4, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4, sampler=val_sampler)

model = RecommendationTransformer(vocab_size=num_item, num_pos=max_len)
model = model.to(device)
model = DDP(module=model, device_ids=[args.gpu])

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


min_val_loss = 1e+9

for epoch in range(args.epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    
    train_loss = []
    for idx, batch in enumerate(tqdm(train_loader)):
        src = batch['source'].to(device)        # batch_size * max_len(157)
        src_mask = batch['source_mask'].to(device)
        tgt = batch['target'].to(device)
        tgt_mask = batch['target_mask'].to(device)

        optimizer.zero_grad()
        
        output = model(src, src_mask)

        loss = calculate_loss(output, tgt, tgt_mask)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        
        if args.rank == 0:
            if idx % 100 == 1:
                wandb.log({'train step loss': np.mean(train_loss[max(0,idx-100):idx])})
                
    model.eval()
    with torch.no_grad():
        val_loss = []
        for _, batch in enumerate(tqdm(val_loader)):
            src = batch['source'].to(device)
            src_mask = batch['source_mask'].to(device)
            tgt = batch['target'].to(device)
            tgt_mask = batch['target_mask'].to(device)

            output = model(src, src_mask)
            
            loss = calculate_loss(output, tgt, tgt_mask)
            val_loss.append(loss.item())
                        
        val_loss = np.mean(val_loss)


    if args.rank == 0:
        wandb.log({"train_loss": np.mean(train_loss), "val_loss": val_loss})

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            
            checkpoint = {'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f"checkpoints/checkpoint.tar")