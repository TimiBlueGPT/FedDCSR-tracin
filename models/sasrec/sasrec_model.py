# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .modules import SelfAttention
from . import config



class SASRec(nn.Module):
    def __init__(self, num_items, args):
        super(SASRec, self).__init__()
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"

        self.item_emb = nn.Embedding(num_items + 1, config.hidden_size,
                                     padding_idx=num_items)
        self.pos_emb = nn.Embedding(args.max_seq_len, config.hidden_size)

        self.encoder = SelfAttention(num_items, args)

        self.linear = nn.Linear(config.hidden_size, num_items)
        self.linear_pad = nn.Linear(config.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)

    def add_position_embedding(self, seqs, seq_embeddings):
        seq_length = seqs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=seqs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(seqs)
        position_embeddings = self.pos_emb(position_ids)
        seq_embeddings += position_embeddings
        seq_embeddings = self.LayerNorm(seq_embeddings)
        seq_embeddings = self.dropout(seq_embeddings)
        return seq_embeddings

    def forward(self, seqs):
        seqs_emb = self.item_emb(seqs)
        seqs_emb = seqs_emb * self.item_emb.embedding_dim ** 0.5
        seqs_emb = self.add_position_embedding(seqs, seqs_emb)

        seqs_fea = self.encoder(seqs_emb, seqs)

        result = self.linear(seqs_fea)
        result_pad = self.linear_pad(seqs_fea)
        return torch.cat((result, result_pad), dim=-1)
