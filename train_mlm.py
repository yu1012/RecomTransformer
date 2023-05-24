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
from constants import TRAIN_CONSTANTS
from utils import *

import wandb

args = parse_args()
init_for_distributed(args)


if args.rank == 0:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RecomTransformer",
        name=f'{args.dataset}',
        # track hyperparameters and run metadata
        config={
        "heads": args.heads,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "epoch": args.epoch,
        "lr": args.lr
        }
    )

feat = torch.load(f'data_prc/embs/{args.dataset}.pt')
feat = F.normalize(feat, dim=1)
feat = torch.cat((torch.zeros(2, feat.shape[-1]), feat), dim=0)

# feat = torch.load(f'data_prc/word2vec/{args.dataset}.pt')
# feat = None

with open(file=f'data_prc/{args.dataset}_data.pkl', mode='rb') as f:
    data = pickle.load(f)

train_session, val_session, test_session = data

num_item = 1+max([max(sess) for sess in train_session+val_session+test_session])
max_len = max([len(sess) for sess in train_session+val_session+test_session])

train_dataset = Bert4RecDataset(train_session, max_len, split_mode="train")
val_dataset = Bert4RecDataset(val_session, max_len, split_mode="val")

train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
# val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4) # , sampler=val_sampler)

device = 'cuda:{}'.format(args.gpu)

feat = feat.to(device)

model = RecommendationTransformer(vocab_size=num_item, heads=args.heads, layers=args.layers, emb_dim=args.dim, num_pos=max_len, feat=feat)
model = model.to(device)
model = DDP(module=model, device_ids=[args.gpu])

optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=1e-8)

min_val_loss = 1e+9
max_mrr = 0

for epoch in range(args.epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    
    train_loss = []
    ind = torch.LongTensor([i for i in range(0, num_item)]).to(device)
    # ind = 1

    for idx, batch in enumerate(tqdm(train_loader)):
        src = batch['source'].to(device)        # batch_size * max_len(10)
        src_mask = batch['source_mask'].to(device)
        tgt = batch['target'].to(device)
        mask = (src == TRAIN_CONSTANTS.MASK)
        
        optimizer.zero_grad()
        
        output = model(src, src_mask, ind)
        loss = calculate_loss(output, tgt, mask)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        
        if args.rank == 0:
            if idx % 100 == 1:
                wandb.log({'train step loss': np.mean(train_loss[max(0,idx-100):idx])})
                
    model.eval()
    with torch.no_grad():
        val_loss = []
        mrr = []
        recall = []
        for _, batch in enumerate(tqdm(val_loader)):
            src = batch['source'].to(device)
            src_mask = batch['source_mask'].to(device)
            tgt = batch['target'].to(device)
            mask = (src == TRAIN_CONSTANTS.MASK)

            output = model(src, src_mask, ind)
            loss = calculate_loss(output, tgt, mask)
            val_loss.append(loss.item())

            mask = mask.nonzero()[:,1].unsqueeze(-1)
            index  = torch.arange(src.shape[0]).unsqueeze(-1)
            output = output[index, :, mask].squeeze()
            tgt = tgt[index, mask].squeeze()

            topk = torch.topk(output, k=110).indices
            for i in range(topk.shape[0]):
                topk_i = topk[i]
                topk_i = topk_i[topk_i != 0]
                topk_i = topk_i[topk_i != 1]
                
                rank = (topk_i[~torch.isin(topk_i, src[i])][:100]==tgt[i]).nonzero()

                if len(rank) != 0:
                    mrr.append(1/(1+rank.cpu().item()))
                    recall.append(1)
                else:
                    mrr.append(0)
                    recall.append(0)

        val_loss = np.mean(val_loss)
        mrr = np.mean(mrr)
        recall = np.mean(recall)


    if args.rank == 0:
        wandb.log({"train_loss": np.mean(train_loss), "val_loss": val_loss, "MRR@100": mrr, "Recall@100": recall, "epoch": epoch})

        if max_mrr < mrr:
            max_mrr = mrr
            
            checkpoint = {'epoch': epoch,
                          'model_state_dict': model.module.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f"checkpoints/{args.dataset}_checkpoint.tar")
