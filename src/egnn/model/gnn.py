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
    def __init__(self, input_node_dim, input_edge_dim, output_dim, attention=False,
                 hidden_dim=64, num_layers=4, predict_type="unembedding"):
        """

        :param input_node_dim:
        :param input_edge_dim:
        :param output_dim:
        :param hidden_dim:
        :param num_layers:
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
    def __init__(self, input_node_dim, input_edge_dim, output_dim, attention=False, num_coords=3,
                 num_velocities=0, update_coords=True, hidden_dim=64, num_layers=4,
                 predict_type="coord"):
        """

        :param input_node_dim:
        :param input_edge_dim:
        :param output_dim:
        :param num_coords:
        :param num_velocities:
        :param update_coords:
        :param hidden_dim:
        :param num_layers:
        :param predict_type:
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
            raise ValueError()

        return pred, edge_indices, edges_features
