import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_layers=3, adj_matrix=None, 
                 dropout=0.5, device="cuda:1"):
        """
        Args:
            input_dim: размерность входных признаков
            hidden_dims: список скрытых размерностей
            output_dim: размерность выходных признаков
            n_layers: количество слоев
            adj_matrix: матрица смежности (если None, будет ожидаться в forward)
            dropout: вероятность dropout
            device: устройство для вычислений
        """
        super(GNN, self).__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        self.dropout = dropout
        
        if adj_matrix is not None:
            self.register_buffer('A', torch.tensor(adj_matrix, dtype=torch.float))
        else:
            self.A = None
        
        for i in range(n_layers):
            use_skip = (i != n_layers - 1)
            self.layers.append(
                GNNLayer(dims[i], dims[i+1], use_skip=use_skip)
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
    def forward(self, X, A=None):
        """
        Args:
            X: начальная матрица признаков узлов [n_nodes, input_dim]
            A: (опционально) матрица смежности [n_nodes, n_nodes]
        Returns:
            X: выходная матрица признаков [n_nodes, output_dim]
        """
        if A is None:
            A = self.A
        
        if A is not None and X.device != A.device:
            X = X.to(A.device)
        
        for i, layer in enumerate(self.layers):
            X = layer(A, X)
            if i != len(self.layers) - 1:  # No ReLU for last layer
                X = F.relu(X)
                X = F.dropout(X, p=self.dropout, training=self.training)
        return X

