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


def get_topX(x):
    '''Get top X% of the values in the matrix'''
    return X * np.array(
        X > np.percentile(X, 85), dtype=int
    )


def get_adj(x):
    '''Get adjacency matrix from the matrix'''
    adj = SparseTensor(
        row=torch.tensor(np.array(x.nonzero()))[0],
        col=torch.tensor(np.array(x.nonzero()))[1],
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    return adj


def get_data(X, metric="linear"):
    '''Get data and adjacency matrix from the matrix'''
    dist = pairwise_kernels(X, metric=metric)
    dist_x = get_topX(dist)
    return torch.tensor(X.values, dtype=torch.float), get_adj(dist_x)


class AE_GCN(Module):
    '''Autoencoder with GCN layers'''
    def __init__(
        self,
        params,
    ):
        super(AE_GCN, self).__init__()
        self.gcn1 = GCNConv(params["hidden1"], params["hidden2"])
        self.gcn2 = GCNConv(params["hidden2"], params["hidden1"])

        self.dropout1 = Dropout(params["dropout1"])
        self.dropout2 = Dropout(params["dropout2"])

        self.graph_norm1 = GraphNorm(params["hidden2"])
        self.graph_norm2 = GraphNorm(params["hidden1"])

        self.gcn3 = GCNConv(params["hidden0"], params["hidden3"])
        self.gcn4 = GCNConv(params["hidden3"], params["hidden0"])

        self.graph_norm3 = GraphNorm(params["hidden3"])
        self.graph_norm4 = GraphNorm(params["hidden0"])

        self.batch_norm1 = BatchNorm1d(params["hidden1"])
        self.batch_norm2 = BatchNorm1d(params["hidden0"])

    def forward(self, data, x, adj, x_t, adj_t):
        '''
        data: gene expression matrix
        x: gene expression matrix
        adj: cell-cell similarity matrix
        x_t: transposed gene expression matrix
        adj_t: gene-gene similarity matrix
        '''
        # For Cell similarity
        x = self.dropout1(relu(self.graph_norm1(self.gcn1(x, adj.t()))))
        x = relu(self.graph_norm2(self.gcn2(x, adj.t())))

        # For Gene similarity
        x_t = self.dropout2(relu(self.graph_norm3(self.gcn3(x_t, adj_t.t()))))
        x_t = relu(self.graph_norm4(self.gcn4(x_t, adj_t.t())))

        res = x + x_t.T

        if params["clustering"]:
            res = self.batch_norm1(data) + self.batch_norm2(data.T).T

        return res


def run_model(input_data, params=None, clustering=False):
    '''Run model

    input_data: gene expression matrix
    params: hyperparameters
    clustering: whether to add batch normalized data
    '''

    params = {
        "device": "cuda",
        "dropout1": 0.3,
        "dropout2": 0.1,
        "epochs": 1500,
        "hidden2": 128,
        "hidden3": 1024,
        "lr": 0.0001,
        "optimizer": "Adam",
        "clustering": False,
    }

    if clustering:
        params = {
            "device": "cuda",
            "dropout1": 0.4,
            "dropout2": 0.3,
            "epochs": 100,
            "hidden2": 256,
            "hidden3": 512,
            "lr": 0.01,
            "optimizer": "Adam",
            "clustering": True,
        }

    x, adj = get_data(input_data)
    x_t, adj_t = get_data(input_data.T)

    x = x.to(params["device"])
    adj = adj.to(params["device"])
    x_t = x_t.to(params["device"])
    adj_t = adj_t.to(params["device"])

    hidden0 = input_data.shape[0]
    hidden1 = input_data.shape[1]

    model = AE_GCN(params).to(params["device"])
    loss_function = MSELoss().to(params["device"])
    optimizer = getattr(torch.optim, params["optimizer"])(
        model.parameters(),
        lr=params["lr"],
    )
    losses = []
    res = pd.DataFrame()

    for epoch in range(params["epochs"]):
        reconstructed = model(x, x, adj, x_t, adj_t)
        loss = loss_function(reconstructed, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    pred = reconstructed.cpu().detach().numpy()
    return pred
