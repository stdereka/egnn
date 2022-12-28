import torch
from .convolution import GraphConvolution, EquivariantGraphConvolution
from .utils import unpack_node_params, pack_node_params


class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(ResidualMLP, self).__init__()
        self.h = hidden_dim
        self.l1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.a1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x_ = self.l1(x)
        x_ = self.a1(x_)
        x_ = self.l2(x_)
        return x[:, :self.h] + x_


class GNN(torch.nn.Module):
    """Graph Convolutional Neural Network"""
    def __init__(self, input_node_dim, input_edge_dim, output_dim, attention=False,
                 hidden_dim=64, num_layers=4, predict_type="unembedding"):
        """
        :param input_node_dim: Number of node features.
        :param input_edge_dim: Number of edge features.
        :param output_dim: How many features to predict for each node.
        :param attention: Whether to use attention for neighbour aggregation.
        :param hidden_dim: Number of features in all hidden layers.
        :param num_layers: Number of convolutional layers.
        :param predict_type: Model prediction type. `unembedding` - predict `output_dim` features for each node.
        `node_pooling` - predict one feature for the entire graph.
        """
        super(GNN, self).__init__()

        self.embedding = torch.nn.Linear(input_node_dim, hidden_dim)
        assert predict_type in ("unembedding", "node_pooling")
        self.predict_type = predict_type
        self.out_dim = output_dim

        convs = []
        for _ in range(num_layers):
            phi_edge = torch.nn.Sequential(*[
                torch.nn.Linear(2 * hidden_dim + input_edge_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU()
            ])

            phi_node = ResidualMLP(hidden_dim)

            if attention:
                phi_attention = torch.nn.Sequential(*[
                    torch.nn.Linear(hidden_dim, 1),
                    torch.nn.Sigmoid()
                ])
            else:
                phi_attention = None

            conv = GraphConvolution(phi_edge, phi_node, phi_attention)
            convs.append(conv)

        self.convs = torch.nn.Sequential(*convs)
        self.unembedding = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        ])

        if self.predict_type == "node_pooling":
            self.pool = torch.nn.Sequential(*[
                torch.nn.Linear(output_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, 1),
            ])

    def forward(self, node_features, edge_indices, edges_features, batch_size):
        """
        :param node_features: Node features, `num_nodes` x `input_node_dim` tensor.
        :param edge_indices: Edge node indices: two tensors (`first_node_ids`, `second_node_ids`)
        :param edges_features: Edge features, `num_edges` x `input_edge_dim` tensor.
        :param batch_size: How many graphs passed in a batch. Required for `node_pooling` prediction type.
        :return: (model prediction, edge_indices, edges_features)
        """
        node_features = self.embedding(node_features)
        node_features, _, _ = self.convs((node_features, edge_indices, edges_features))

        if self.predict_type == "unembedding":
            pred = self.unembedding(node_features)
        elif self.predict_type == "node_pooling":
            node_features = self.unembedding(node_features)
            node_features = node_features.view(-1, len(node_features) // batch_size, self.out_dim)
            node_features = node_features.sum(dim=1)
            pred = self.pool(node_features)
        else:
            raise ValueError("Unknown predict type.")

        return pred, edge_indices, edges_features


class EquivariantGNN(torch.nn.Module):
    """E(n) Equivariant Graph Neural Network."""
    def __init__(self, input_node_dim, input_edge_dim, output_dim, attention=False, num_coords=3,
                 num_velocities=0, update_coords=True, hidden_dim=64, num_layers=4,
                 predict_type="coord"):
        """
        :param input_node_dim: Number of node features.
        :param input_edge_dim: Number of edge features.
        :param output_dim: How many features to predict for each node.
        :param attention: Whether to use attention for neighbour aggregation.
        :param num_coords: Number of coordinate features. First `num_coords` features in `node_features` are treated as
        coordinates.
        :param num_velocities: Number of vector "velocity" features.
        Features from `num_coords` + 1 to `num_coords` + `num_velocities` in `node_features` are treated as
        velocities.
        :param update_coords: Whether to update coordinated on each convolution layer.
        :param hidden_dim: Number of features in all hidden layers.
        :param num_layers: Number of convolutional layers.
        :param predict_type: Model prediction type. `unembedding` - predict `output_dim` features for each node.
        `node_pooling` - predict one feature for the entire graph. `coord` - return final coordinates for each node.
        """
        super(EquivariantGNN, self).__init__()

        self.num_coords = num_coords
        self.num_velocities = num_velocities

        assert predict_type in ("coord", "unembedding", "node_pooling")
        self.predict_type = predict_type
        self.out_dim = output_dim

        inp_dim = input_node_dim - num_coords - num_velocities + int(num_velocities != 0)

        self.embedding = torch.nn.Linear(inp_dim, hidden_dim)

        convs = []
        for _ in range(num_layers):
            phi_edge = torch.nn.Sequential(*[
                torch.nn.Linear(2 * hidden_dim + 1 + input_edge_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU()
            ])

            phi_node = ResidualMLP(hidden_dim)

            phi_coords = torch.nn.Sequential(*[
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, output_dim),
            ])

            if self.num_velocities:
                phi_velocity = torch.nn.Sequential(*[
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 1),
                ])
            else:
                phi_velocity = None

            if attention:
                phi_attention = torch.nn.Sequential(*[
                    torch.nn.Linear(hidden_dim, 1),
                    torch.nn.Sigmoid()
                ])
            else:
                phi_attention = None

            conv = EquivariantGraphConvolution(phi_edge, phi_node, phi_coords, phi_velocity, phi_attention,
                                               num_coords=num_coords,
                                               num_velocities=num_velocities,
                                               update_coords=update_coords)
            convs.append(conv)

        self.convs = torch.nn.Sequential(*convs)
        self.unembedding = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        ])

        if self.predict_type == "node_pooling":
            self.pool = torch.nn.Sequential(*[
                torch.nn.Linear(output_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, 1),
            ])

    def forward(self, node_features, edge_indices, edges_features, batch_size):
        """
        :param node_features: Node features, `num_nodes` x `input_node_dim` tensor.
        :param edge_indices: Edge node indices: two tensors (`first_node_ids`, `second_node_ids`)
        :param edges_features: Edge features, `num_edges` x `input_edge_dim` tensor.
        :param batch_size: How many graphs passed in a batch. Required for `node_pooling` prediction type.
        :return: (model prediction, edge_indices, edges_features)
        """
        coords, velocities, node_features = unpack_node_params(node_features, self.num_coords, self.num_velocities)
        node_features = self.embedding(node_features)

        node_features, _, _ = self.convs(
            (pack_node_params(coords, velocities, node_features),
             edge_indices, edges_features)
        )

        coords, _, node_features = unpack_node_params(node_features, self.num_coords, self.num_velocities)

        if self.predict_type == "coord":
            pred = coords
        elif self.predict_type == "unembedding":
            pred = self.unembedding(node_features)
        elif self.predict_type == "node_pooling":
            node_features = self.unembedding(node_features)
            node_features = node_features.view(-1, len(node_features) // batch_size, self.out_dim)
            node_features = node_features.sum(dim=1)
            pred = self.pool(node_features)
        else:
            raise ValueError("Unknown predict type.")

        return pred, edge_indices, edges_features
