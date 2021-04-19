import os

import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv

from conv_layers import NormGCNConv
from environment_setup import PROJECT_ROOT_DIR
import torch.nn as nn

# module = GATConv
module = ChebConv
# module = GCNConv
# module = NormGCNConv
bias = False


class BaseModel(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.save_path = os.path.join(PROJECT_ROOT_DIR, 'checkpoints')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.vars = {}

        self.outputs = None

    def _build(self):
        raise NotImplementedError

    def predict(self, *args):
        pass

    def save(self):
        torch.save(self.state_dict(), f"{self.save_path}/%s.ckpt" % self.name)
        print("Model saved in file: %s" % self.save_path)

    def load(self):
        save_path = f"{self.save_path}/%s.ckpt" % self.name
        self.load_state_dict(torch.load(save_path))
        print("Model restored from file: %s" % save_path)


class MLP(nn.Module, BaseModel):
    def __init__(self, input_dim, hidden1, dropout=0, **kwargs):
        super(MLP, self).__init__()
        super(nn.Module, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden1 = hidden1
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = 2
        self.dropout = dropout
        self._build()

    def _build(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden1, self.output_dim),
            nn.Dropout(self.dropout)
        )

    def predict(self, inputs):
        self.outputs = self.layers(inputs)
        return self.outputs


class GCN(nn.Module, BaseModel):
    def __init__(self, input_dim, hidden1, dropout=0, degree=4, **kwargs):
        super(GCN, self).__init__()
        super(nn.Module, self).__init__(**kwargs)
        self.hidden1 = hidden1
        self.input_dim = input_dim
        self.output_dim = 2
        self.dropout = dropout
        self.degree = degree
        self._build()

    def _build(self):
        self.layers = nn.Sequential(
            module(in_channels=self.input_dim,
                   out_channels=self.hidden1,
                   K=self.degree,
                   bias=bias),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            module(in_channels=self.hidden1,
                   out_channels=self.output_dim,
                   K=self.degree,
                   bias=bias)
        )

    def predict(self, inputs, edge_index, edge_attr=None):
        for f in self.layers:
            if isinstance(f, module):
                inputs = f(inputs, edge_index, edge_attr)
            else:
                inputs = f(inputs)
        self.outputs = inputs
        return self.outputs

    def __repr__(self):
        return f"The GCN has :- {self.hidden1, self.input_dim, self.output_dim, self.dropout, self.degree}"


class Deep_GCN(GCN):
    def __init__(self, input_dim, hidden1, depth, dropout=0, degree=4, **kwargs):
        self.depth = depth
        self.degree = degree
        super(Deep_GCN, self).__init__(input_dim, hidden1, dropout=dropout, **kwargs)

    def _build(self):
        layer_list = nn.ModuleList()
        layer_list.extend(
            [
                module(in_channels=self.input_dim,
                       out_channels=self.hidden1,
                       K=self.degree,
                       bias=bias),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
        )
        for _ in range(self.depth):
            layer_list.extend(
                [
                    module(in_channels=self.hidden1,
                           out_channels=self.hidden1,
                           K=self.degree,
                           bias=bias),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)]
            )

        layer_list.extend(
            [
                module(in_channels=self.hidden1,
                       out_channels=self.output_dim,
                       K=self.degree,
                       bias=bias),
            ]
        )
        self.layers = nn.Sequential(*layer_list)


if __name__ == '__main__':
    features = torch.randn(8, 2000)
    input_dim = 2000
    hidden1 = 32
    depth = 5
    mlp = MLP(input_dim=input_dim, hidden1=hidden1)
    print(mlp.predict(features).shape)
    dgcn = Deep_GCN(input_dim=input_dim, hidden1=hidden1, depth=depth)
    adj = torch.as_tensor([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    print(list(zip(torch.nonzero(adj, as_tuple=False))))
    edge_index = torch.as_tensor(
        [[int(e[0]), int(e[1])] for e in torch.nonzero(adj, as_tuple=False)],
        dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_index[0, :])
    print(dgcn.predict(features, edge_index, None).shape)
    dgcn.save()
    dgcn.load()
