import torch.nn as nn

from layers import ReversibleInstanceNormalization as RIN
from layers import HyperLinearLayer


class InverseTransformer(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, n_input, n_heads=4, n_blocks=2):
        super().__init__()
        self.instance_norm = RIN(n_input)
        self.embedding = nn.Linear(d_input, d_hidden) # d_input > d_model
        tx_enc_block = nn.TransformerEncoderLayer(d_hidden, nhead=n_heads, dim_feedforward=d_hidden*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(tx_enc_block, num_layers=n_blocks)
        self.projection = nn.Linear(d_hidden, d_output) # d_model > d_output
    
    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.embedding(X)
        X = self.transformer(X)
        X = self.projection(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X
    

class InverseTransformerHyper(nn.Module):

    def __init__(self, d_input, d_output, d_hidden, d_hyper_hidden, n_input, d_embedding=8, embedding=None, n_heads=8, n_blocks=3):
        super().__init__()
        self.n_input = n_input
        self.d_output = d_output
        self.instance_norm = RIN(n_input)
        self.embedding = nn.Linear(d_input, d_hidden) # d_input > d_model
        tx_enc_block = nn.TransformerEncoderLayer(d_hidden, nhead=n_heads, dim_feedforward=d_hidden*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(tx_enc_block, num_layers=n_blocks)
        self.projection_hyper = HyperLinearLayer(d_hidden, d_output, d_hyper_hidden, n_input, d_embedding, embedding)
    
    def forward(self, X):
        X = self.instance_norm(X.transpose(1, 2), "norm").transpose(1, 2)
        X = self.embedding(X)
        X = self.transformer(X)
        X = self.projection_hyper(X)
        X = self.instance_norm(X.transpose(1, 2), "denorm").transpose(1, 2)
        return X