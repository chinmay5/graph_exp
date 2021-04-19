import torch
import torch_geometric.nn as gnn
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor, matmul


class NormGCNConv(gnn.conv.GCNConv):

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NormGCNConv, self).__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias,
                                          **kwargs)

    def forward(self, x, edge_index, edge_weight=None, add_loops=False):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return super().forward(x=x, edge_index=edge_index, edge_weight=edge_weight)


if __name__ == '__main__':
    conv = NormGCNConv(in_channels=2000, out_channels=2)
    features = torch.randn(8, 2000)
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
    edge_index = torch.as_tensor(
        [[int(e[0]), int(e[1])] for e in torch.nonzero(adj, as_tuple=False)],
        dtype=torch.long).t().contiguous()
    print(conv(features, edge_index, add_loops=True).shape)
