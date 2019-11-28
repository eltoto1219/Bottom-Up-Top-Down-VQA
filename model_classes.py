import torch 
import torch.nn as nn 
import torch.nn.functional as F
from utils import load_weights, mask_tensor

class CoAttention(nn.Module):
    def __init__(self, v_emb, q_emb, proj_dim, dropout=0.2, bn = False, wn = False, ln = False):
        super().__init__()
        self.v_proj = MLP(v_emb, proj_dim, bn = bn, wn = wn, ln = ln, img = True)
        self.q_proj = MLP(q_emb, proj_dim, bn = bn, wn = wn, ln = ln)
        self.linear = nn.Linear(proj_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, v, n_objs):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)
        q_proj = self.q_proj(q)
        q_proj = q_proj.unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        logits = self.dropout(self.linear(joint_repr))
        logits = mask_tensor(logits, n_objs)
        logits[logits == 0] = float("-inf")
        probs = F.softmax(logits, dim = 1)
        return probs

class Embedding(nn.Module):
    def __init__(self, weight_dir, pad_idx = 0):
        super().__init__()

        self.pad_idx = pad_idx
        weight_matrix = load_weights(weight_dir)
        weight_matrix = torch.tensor(weight_matrix)         
        self.n_tokens = weight_matrix.size(0)
        self.emb_dim = weight_matrix.size(1)                 
        self.embed = nn.Embedding(self.n_tokens , self.emb_dim , padding_idx = pad_idx )
        self.embed.weight = nn.Parameter(weight_matrix)
        self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)
        return x

class Encoder(nn.Module):
    def __init__(self, emb_dim, h_dim, n_layers, dropout, bidir = True, max_len = 14, 
        bn = False, 
        wn = False, 
        ln = False):
        super().__init__()

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.dropout = nn.Dropout(dropout)
        self.bidir = bidir
        self.h0 = nn.Parameter(torch.zeros(1), requires_grad = False)
        self.c0 = nn.Parameter(torch.zeros(1), requires_grad = False)
        self.rnn = nn.LSTM(emb_dim, self.h_dim, n_layers, dropout = dropout, bidirectional = bidir)
        if self.bidir:
            #self.bi_combine = MLP(self.h_dim *2, self.h_dim, bn = bn, wn = wn, ln = ln)
            self.bi_combine = nn.Linear(self.h_dim *2, self.h_dim)
            
    def init_hidden(self, embeddings):
        batch = embeddings.size(0)  
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch, self.h_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers, batch, self.h_dim)
        return h0, c0

    def forward(self, embeddings, in_lengths):

        (h0, c0) = self.init_hidden(embeddings)

        embeddings = embeddings.permute(1,0,2)

        embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, 
            lengths = in_lengths, 
            enforce_sorted = False)

        self.rnn.flatten_parameters()

        packed_outputs, (hidden, cell) = self.rnn(embeddings, (h0, c0))
        
        cell = self.bi_combine(torch.cat((cell[-2], cell[-1]), dim =-1)) if self.bidir else cell[-1]
        hidden = self.bi_combine(torch.cat((hidden[-2], hidden[-1]), dim =-1))\
            if self.bidir else hidden[-1]

        outputs, lengths = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, 
            total_length = self.max_len)
        outputs = outputs.permute(1,0,2)

        return cell, hidden, outputs

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, wn  = False, bn = False, ln = False, img = False, 
        classify = False, 
        dropout = 0.2,  
        max_objs = 100):
        super().__init__()  

        self.classify = classify
        self.linear_1 = nn.Linear(in_dim, in_dim * 2)
        self.linear_2 = nn.Linear(in_dim * 2, out_dim)

        self.norm = ln or bn
        if img and bn:
           self.norm1 = nn.BatchNorm1d(max_objs)
           self.norm2 = nn.BatchNorm1d(max_objs)
        elif bn:
           self.norm1 = nn.BatchNorm1d(in_dim * 2)
           self.norm2 = nn.BatchNorm1d(out_dim)
        elif ln:
           self.norm1 = nn.LayerNorm(in_dim * 2)
           self.norm2 = nn.LayerNorm(out_dim)
        elif wn:
           self.linear_1 = nn.utils.weight_norm(self.linear_1)
           self.linear_2 = nn.utils.weight_norm(self.linear_2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(F.relu(self.linear_1(x))) if self.norm else self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(self.norm2(F.relu(x))) if not self.classify and self.norm else x
        #x = self.dropout(F.relu(x)) if not self.classify else x
        return x
