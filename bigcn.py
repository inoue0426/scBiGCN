import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics.pairwise import pairwise_kernels
from torch.nn import BatchNorm1d, Dropout, Linear, Module, MSELoss
from torch.nn.functional import relu
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphNorm
from torch_sparse import SparseTensor
from tqdm import tqdm


def get_topX(X):
    return X * np.array(
        X > max(np.mean(X), np.median(X), np.percentile(X, 85)), dtype=int
    )


def get_adj(x):
    adj = SparseTensor(
        row=torch.tensor(np.array(x.nonzero()))[0],
        col=torch.tensor(np.array(x.nonzero()))[1],
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    return adj


def get_data(X, metric="linear"):
    dist = pairwise_kernels(X, metric=metric)
    dist_x = get_topX(dist)
    return torch.tensor(X.values, dtype=torch.float), get_adj(dist_x)


class AE_GCN(Module):
    def __init__(self, params,):
        super(AE_GCN, self).__init__()
        self.gcn1 = GCNConv(params["hidden1"], params["hidden2"])
        self.gcn4 = GCNConv(params["hidden2"], params["hidden1"])

        self.dropout1 = Dropout(params["dropout1"])

        self.graph_norm1 = GraphNorm(params["hidden2"])
        self.graph_norm4 = GraphNorm(params["hidden1"])

        self.gcn5 = GCNConv(params["hidden0"], params["hidden7"])
        self.gcn8 = GCNConv(params["hidden7"], params["hidden0"])

        self.dropout4 = Dropout(params["dropout4"])

        self.graph_norm5 = GraphNorm(params["hidden7"])
        self.graph_norm8 = GraphNorm(params["hidden0"])

        self.batch_norm1 = BatchNorm1d(params["hidden1"])
        self.batch_norm2 = BatchNorm1d(params["hidden0"])

    def forward(self, data, x, adj, x_t, adj_t, clustering):
        # For Cell similarity
        x = self.dropout1(relu(self.graph_norm1(self.gcn1(x, adj.t()))))
        x = relu(self.graph_norm4(self.gcn4(x, adj.t())))

        # For Gene similarity
        x_t = self.dropout4(relu(self.graph_norm5(self.gcn5(x_t, adj_t.t()))))
        x_t = relu(self.graph_norm8(self.gcn8(x_t, adj_t.t())))

        res = x + x_t.T

        if clustering:
            res = self.batch_norm1(data) + self.batch_norm2(data.T).T

        return res


def run_model(input_data, params):
    x, adj = get_data(input_data)
    x_t, adj_t = get_data(input_data.T)

    x = x.to(params['device'])
    adj = adj.to(params['device'])
    x_t = x_t.to(params['device'])
    adj_t = adj_t.to(params['device'])

    hidden0 = input_data.shape[0]
    hidden1 = input_data.shape[1]

    model = AE_GCN(params).to(params['device'])
    loss_function = MSELoss().to(params['device'])
    optimizer = getattr(torch.optim, params['optimizer_name'])(
        model.parameters(),
        lr=params["lr"],
    )
    losses = []
    res = pd.DataFrame()

    for epoch in range(params['epochs']):
        reconstructed = model(x, x, adj, x_t, adj_t)
        loss = loss_function(reconstructed, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    pred = reconstructed.cpu().detach().numpy()
    return pred
