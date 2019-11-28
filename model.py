import torch 
import torch.nn as nn
import torch.nn.functional as F
from model_classes import Embedding, Encoder, MLP, CoAttention

class Model(nn.Module):
    def __init__(self, 
        weight_dir,
        v_dim = 2048, 
        q_dim = 1024, 
        a_dim = 3133, 
        common_dim = 1024, 
        enc_layers = 2, 
        glove_dim = 300, 
        bn = False, 
        wn = False, 
        ln = False):

        super().__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.common_dim = common_dim
        self.enc_layers = enc_layers
        self.glove_dim = glove_dim
 
        self.embedding = Embedding(
                weight_dir)
        self.encoder = Encoder(
                glove_dim, 
                q_dim, 
                enc_layers, 
                bidir = True, 
                bn = bn, 
                ln = ln,
                wn = wn,
                dropout = 0.00) 
        self.attention = CoAttention(
                v_dim, 
                q_dim, 
                common_dim,
                bn = bn, 
                ln = ln,
                wn = wn)
        self.q_proj = MLP(
                q_dim,     
                common_dim, 
                bn = bn, 
                ln = ln,
                wn = wn)
        self.v_proj = MLP(
                v_dim,     
                common_dim,
                bn = bn, 
                ln = ln,
                wn = wn)
        self.classifier = MLP(
                common_dim,   
                a_dim, 
                dropout = 0.5, 
                classify = True,
                bn = bn, 
                ln = ln,
                wn = wn)

    def forward(self, question, q_lens, features, n_objs):
        embeddings = self.embedding(question)
        cell, hidden, outputs = self.encoder(embeddings, q_lens)
        att = self.attention(cell, features, n_objs)
        att_features = att * features
        v = self.v_proj(att_features.sum(1))
        q = self.q_proj(cell)
        joint_repr = q * v
        pred  = self.classifier(joint_repr)
        return pred

    def grad_params(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p 
