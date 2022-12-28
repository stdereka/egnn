import torch

from .utils import unsorted_segment_sum, unsorted_segment_mean, unpack_node_params, pack_node_params


class GraphConvolution(torch.nn.Module):
    def __init__(self, phi_edge, phi_node, phi_attention=None):
        super(GraphConvolution, self).__init__()
        self.phi_edge = phi_edge
        self.phi_node = phi_node
        self.phi_attention = phi_attention

    def forward(self, x):
        node_features, edge_indices, edges_features = x
        first, second = edge_indices

        # Equation 2.1
        edges = torch.cat([node_features[first], node_features[second], edges_features], dim=1)
        edges = self.phi_edge(edges)

        if self.phi_attention:
            edges_attn = self.phi_attention(torch.abs(node_features[first] - node_features[second]))
            edges = edges_attn * edges

        # Equation 2.2
        neighbours_sum = unsorted_segment_sum(edges, first, node_features.size()[0])
        nodes = torch.cat([node_features, neighbours_sum], dim=1)

        # Equation 2.3
        nodes = self.phi_node(nodes)

        return nodes, edge_indices, edges_features


class EquivariantGraphConvolution(torch.nn.Module):
    def __init__(self, phi_edge, phi_node, phi_coord, phi_velocity=None,
                 phi_attention=None, num_coords=3, num_velocities=0, update_coords=True):
        super(EquivariantGraphConvolution, self).__init__()
        self.phi_edge = phi_edge
        self.phi_node = phi_node
        self.phi_coord = phi_coord
        self.phi_velocity = phi_velocity
        self.phi_attention = phi_attention
        self.num_coords = num_coords
        self.num_velocities = num_velocities
        self.update_coords = update_coords

    def forward(self, x):
        node_features, edge_indices, edges_features = x
        coords, velocities, node_features = unpack_node_params(node_features, self.num_coords, self.num_velocities)
        if coords is None:
            raise ValueError("EGNN requires coordinates.")
        first, second = edge_indices

        # Compute coordinate terms required for equations 3 and 4
        coords_diff = coords[first] - coords[second]
        coords_diff_norm = torch.sum(coords_diff ** 2, 1).unsqueeze(1)

        # Page 2 https://arxiv.org/pdf/2102.09844v3.pdf
        # Equation 3
        edges = torch.cat([node_features[first], node_features[second], coords_diff_norm, edges_features], dim=1)
        edges = self.phi_edge(edges)

        # Equation 4
        if self.update_coords:
            coords = coords + unsorted_segment_mean(coords_diff * self.phi_coord(edges), first, coords.size()[0])
            if self.phi_velocity:
                coords = coords + self.phi_velocity(node_features) * velocities

        # Equation 5 (same as in GraphConvolutionLayer)
        if self.phi_attention:
            edges_attn = self.phi_attention(torch.abs(node_features[first] - node_features[second]))
            edges = edges_attn * edges
        neighbours_sum = unsorted_segment_sum(edges, first, node_features.size()[0])

        # Equation 6 (same as in GraphConvolutionLayer)
        nodes = torch.cat([node_features, neighbours_sum], dim=1)
        nodes = self.phi_node(nodes)

        return pack_node_params(coords, velocities, nodes), edge_indices, edges_features
