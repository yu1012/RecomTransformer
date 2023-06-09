import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import EncoderLayer, DecoderLayer
from embeddings import PositionalEncoding

from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dim, dim)

    def forward(self, x, edge_idx, edge_weight):
        x = self.conv1(x, edge_idx, edge_weight)
        x = F.relu(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 emb_dim,
                 layers,
                 heads,
                 dim_key,
                 dim_value,
                 dim_model,
                 dim_inner,
                 pad_id,
                 dropout=0.1,
                 num_pos=200, feat=None, edge_index=None, edge_weight=None, ind=None):

        super().__init__()

        dim = emb_dim
        if feat is not None:
            dim = emb_dim//2

        self.word_embedding = nn.Embedding(source_vocab_size,
                                           dim,
                                           padding_idx=pad_id)
        self.position_encoding = PositionalEncoding(emb_dim, num_pos=num_pos)

        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_model,
                         dim_inner,
                         heads,
                         dim_key,
                         dim_value,
                         dropout=dropout) for _ in range(layers)
        ])

        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        
        self.feat = feat
        self.elu = nn.ELU()
        
        if feat is not None:
            self.feat_transform = nn.Linear(feat.shape[-1], emb_dim//2)
        
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        if self.edge_index is not None:
            self.gnn = GNN(emb_dim)
            self.ind = ind

    def forward(self, source_seq, source_mask, return_attentions=False):

        encoder_self_attention_list = []
        
        if self.feat is not None:
            encoder_output = self.dropout(
                self.position_encoding(torch.cat((self.word_embedding(source_seq), self.elu(self.feat_transform(self.feat[source_seq]))), dim=-1)))
        else:
            encoder_output = self.dropout(
                self.position_encoding(self.word_embedding(source_seq)))
            
        if self.edge_index is not None:
            if self.feat is not None:
                item_feat = torch.cat((self.word_embedding(source_seq), self.elu(self.feat_transform(self.feat[source_seq]))), dim=-1)
            else:
                item_feat = self.word_embedding(self.ind)
        
            item_feat = self.gnn(item_feat, self.edge_index, self.edge_weight)
            encoder_output = self.dropout(self.position_encoding(item_feat[source_seq]))
        

        for encoder_layer in self.layer_stack:

            encoder_output, encoder_self_attention = encoder_layer(
                encoder_output, self_attention_mask=source_mask)

            encoder_self_attention_list += [encoder_self_attention
                                            ] if return_attentions else []

        encoder_output = self.layer_norm(encoder_output)

        if return_attentions:

            return encoder_output, encoder_self_attention_list

        return encoder_output


class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 emb_dim,
                 layers,
                 heads,
                 dim_key,
                 dim_value,
                 dim_model,
                 dim_inner,
                 pad_id,
                 dropout=0.1,
                 num_pos=200):

        super().__init__()

        self.word_embedding = nn.Embedding(target_vocab_size,
                                           emb_dim,
                                           padding_idx=pad_id)
        self.position_encoding = PositionalEncoding(emb_dim, num_pos=num_pos)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(dim_model,
                         dim_inner,
                         heads,
                         dim_key,
                         dim_value,
                         dropout=dropout) for _ in range(layers)
        ])

        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self,
                target_seq,
                target_mask,
                encoder_output,
                source_mask,
                return_attentions=False):

        decoder_self_attention_list = []
        decoder_encoder_attention_list = []

        decoder_output = self.dropout(
            self.position_encoding(self.word_embedding(target_seq)))

        for decoder_layer in self.layer_stack:

            decoder_output, decoder_self_attention, decoder_encoder_attention = decoder_layer(
                decoder_output, encoder_output, target_mask, source_mask)

            decoder_self_attention_list += [decoder_self_attention
                                            ] if return_attentions else []
            decoder_encoder_attention_list += [decoder_encoder_attention
                                               ] if return_attentions else []

        decoder_output = self.layer_norm(decoder_output)

        if return_attentions:

            return decoder_output, decoder_self_attention_list, decoder_encoder_attention_list

        return decoder_output