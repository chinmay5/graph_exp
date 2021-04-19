import torch
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import distance
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch_geometric.transforms as T


def create_pytorch_dataset(final_graph, x_data, y_data):
    x = torch.as_tensor(x_data, dtype=torch.float)
    edge_index = torch.as_tensor(
        [[int(e[0]), int(e[1])] for e in zip(*final_graph.nonzero())],
        dtype=torch.long)
    edge_features = [[final_graph[int(e[0])][int(e[1])]] for e in zip(*final_graph.nonzero())]
    edge_features = torch.as_tensor(np.concatenate(edge_features), dtype=torch.float)  # .unsqueeze(1)
    y_data = torch.as_tensor(y_data, dtype=torch.long).squeeze()
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y_data, edge_attr=edge_features)
    return data


class ABIDEDataset(Dataset):

    def __init__(self, final_graph, x_data, y_data):
        super(Dataset, self).__init__()
        self.final_graph = final_graph
        self.x_data = x_data
        self.y_data = y_data
        self.transform = T.ToSparseTensor(remove_edge_index=False)

    def __len__(self):
        return 1  # We have a single graph here

    def __getitem__(self, index):
        x = torch.as_tensor(self.x_data, dtype=torch.float)
        edge_index = torch.as_tensor(
            [[int(e[0]), int(e[1])] for e in zip(*self.final_graph.nonzero())],
            dtype=torch.long)
        edge_features = [[self.final_graph[int(e[0])][int(e[1])]] for e in zip(*self.final_graph.nonzero())]
        edge_features = torch.as_tensor(np.concatenate(edge_features), dtype=torch.float)
        y_data = torch.as_tensor(self.y_data, dtype=torch.long).squeeze()
        data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y_data, edge_attr=edge_features)
        data = self.transform(data=data)
        return data


if __name__ == '__main__':
    m_x = np.random.randn(4, 5).astype(np.int)
    m_y = torch.as_tensor(np.random.binomial(size=(4, 1), n=1, p=0.5).ravel())
    m_x_dist = distance.pdist(m_x, metric='euclidean')
    m_dist_graph = distance.squareform(m_x_dist)
    m_final_graph = np.where(m_dist_graph > 1, m_dist_graph, 0)
    print(m_final_graph)
    # data = create_pytorch_dataset(final_graph=final_graph, x_data=x, y_data=y)
    # print(data)
    m_dataset = ABIDEDataset(final_graph=m_final_graph, x_data=m_x, y_data=m_y)
    m_data = m_dataset[0]
    # inv_transform = T.ToDense()
    # print(data)
    # data = inv_transform(data)
    print(m_data)
    m_graph = to_networkx(data=m_data, edge_attrs=['edge_attr'], to_undirected=True)
    # nx.draw(graph, with_labels=False, font_weight='bold', node_color=data.y, cmap="Set2")
    m_labels = nx.get_edge_attributes(m_graph, 'edge_attr')
    m_pos = nx.spring_layout(m_graph)
    nx.draw(m_graph, pos=m_pos, with_labels=True)
    nx.draw_networkx_edge_labels(m_graph, m_pos, edge_labels=m_labels)
    plt.savefig('./sample.png')
