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
    HISTORY = 120
    VOCAB_SIZE = 59049
    HEADS = 4
    LAYERS = 6
    EMB_DIM = 256
    DATA_PATH = ""


class Bert4RecDataset(Dataset):
    def __init__(self,
                 sessions,
                 split_mode
                 ):

        super().__init__()

        self.split_mode = split_mode

        self.sessions = sessions
        self.max_len = max([len(sess) for sess in self.sessions])

        self.pad = CONSTANTS.PAD
        self.mask = CONSTANTS.MASK

    def pad_sequence(self, tokens, padding_mode):
        if len(tokens) < self.max_len:
            tokens = tokens + [self.pad] * (self.max_len - len(tokens))
        return tokens

    def mask_sequence(self, sequence, p=0.8):
        return [
            s if random.random() < p else CONSTANTS.MASK
            for s in sequence
        ]

    def mask_last_elements_sequence(self, sequence):
        last = len(sequence)-1
        sequence = sequence[:last] + self.mask_sequence(
            sequence[last:], p=1.0)
        return sequence

    def get_item(self, idx):
        trg_items = self.sessions[idx]

        if self.split_mode == "train":
            src_items = self.mask_sequence(trg_items)
        else:
            src_items = self.mask_last_elements_sequence(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = self.pad_sequence(trg_items, pad_mode)
        src_items = self.pad_sequence(src_items, pad_mode)

        trg_mask = [1 if t != CONSTANTS.PAD else 0 for t in trg_items]
        src_mask = [1 if t != CONSTANTS.PAD else 0 for t in src_items]

        src_items = torch.IntTensor(src_items)
        trg_items = torch.LongTensor(trg_items)
        src_mask = torch.IntTensor(src_mask)
        trg_mask = torch.IntTensor(trg_mask)
        
        return {
            "source": src_items,
            "target": trg_items,
            "source_mask": src_mask,
            "target_mask": trg_mask
        }


    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        return self.get_item(index)