import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random

class CONSTANTS:
    """PAD and MASK token to index
    """
    PAD = 0
    MASK = 1

class Bert4RecDataset(Dataset):
    def __init__(self,
                 sessions,
                 max_len,
                 split_mode
                 ):

        super().__init__()

        self.split_mode = split_mode

        self.sessions = sessions
        self.max_len = max_len

        self.pad = CONSTANTS.PAD
        self.mask = CONSTANTS.MASK

    def pad_sequence(self, tokens, padding_mode):
        if len(tokens) < self.max_len:
            tokens = tokens + [self.pad] * (self.max_len - len(tokens))
        return tokens

    def mask_last_elements_sequence(self, sequence):
        last = len(sequence)-1
        sequence = sequence[:last]
        return sequence

    def get_item(self, idx):
        trg_items = self.sessions[idx][:-1]
        tgt = [trg_items[-1]]

        if self.split_mode == "train":
            src_items = self.mask_last_elements_sequence(trg_items)
        else:
            src_items = self.mask_last_elements_sequence(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = self.pad_sequence(trg_items, pad_mode)
        src_items = self.pad_sequence(src_items, pad_mode)

        trg_mask = [1 if t != CONSTANTS.PAD else 0 for t in trg_items]
        src_mask = [1 if t != CONSTANTS.PAD else 0 for t in src_items]

        src_items = torch.LongTensor(src_items)
        trg_items = torch.LongTensor(trg_items)
        src_mask = torch.IntTensor(src_mask)
        trg_mask = torch.IntTensor(trg_mask)
        
        tgt = torch.LongTensor(tgt)
        mask_token = (src_items==CONSTANTS.PAD).nonzero()[0]-1 if len((src_items==CONSTANTS.PAD).nonzero())!=0 else torch.LongTensor([self.max_len-1])
        
        return {
            "source": src_items,
            "target": tgt,
            "source_mask": src_mask,
            "target_mask": trg_mask,
            "mask_token": mask_token
        }

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        return self.get_item(index)