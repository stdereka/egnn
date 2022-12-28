import torch

from src.egnn.dataset import NBodyDataset
from src.egnn.model import GraphConvolution


def test_graph_convolution():
    phi_edge = torch.nn.Identity()
    phi_node = torch.nn.Identity()
    gcl = GraphConvolution(phi_edge, phi_node)

    nodes = torch.tensor([[0], [1], [2]])
    edge_ids = torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1, 2, 0, 2, 0, 1])
    edge_features = torch.tensor([[0.1], [0.2], [0.1], [1.2], [0.2], [1.2]])

    res = gcl((nodes, edge_ids, edge_features))[0]
    true_res = torch.tensor([[0, 0, 3, 0.3], [1, 2, 2, 1.3], [2, 4, 1, 1.4]])
    assert torch.allclose(res, true_res)


def test_nbody_dataset():
    data = NBodyDataset("/home/caladrius/gitCloned/egnn/n_body_system/dataset/", num_samples=1000)
    assert len(data) == 1000

    graph, label = data[100]
    assert torch.allclose(torch.tensor(label.size()), torch.tensor([5, 3]))

    nodes, edges_ids, edges = graph
    assert torch.allclose(torch.tensor(nodes.size()), torch.tensor([5, 6]))
    assert torch.allclose(torch.tensor(edges_ids[0].size()), torch.tensor([20]))
    assert torch.allclose(torch.tensor(edges.size()), torch.tensor([20, 1]))
