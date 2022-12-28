import os
import torch
import numpy as np


class NBodyDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", time_start=30, time_end=40, num_samples=3000):
        assert split in ("train", "valid", "test")

        loc = np.load(os.path.join(root, f"loc_{split}_charged5_initvel1small.npy"))
        vel = np.load(os.path.join(root, f"vel_{split}_charged5_initvel1small.npy"))
        edges = np.load(os.path.join(root, f"edges_{split}_charged5_initvel1small.npy"))

        loc = np.swapaxes(loc, 2, 3)
        vel = np.swapaxes(vel, 2, 3)

        first = []
        second = []
        edge_features = []

        for i in range(loc.shape[2]):
            for j in range(loc.shape[2]):
                if i != j:
                    first.append(i)
                    second.append(j)
                    edge_features.append(edges[:num_samples, i, j])

        edge_features = np.vstack(edge_features).T
        edges = [torch.tensor(first), torch.tensor(second)]

        self.init_loc = torch.tensor(loc[:num_samples, time_start, :, :])
        self.final_loc = torch.tensor(loc[:num_samples, time_end, :, :])
        self.init_vel = torch.tensor(vel[:num_samples, time_start, :, :])
        self.final_vel = torch.tensor(vel[:num_samples, time_end, :, :])
        self.edges = edges
        self.edge_features = torch.tensor(edge_features).reshape(len(self.init_loc), -1, 1)

    def __getitem__(self, item):
        data = torch.hstack((self.init_loc[item, :], self.init_vel[item, :])), self.edges, self.edge_features[item]
        label = self.final_loc[item, :]
        return data, label

    def __len__(self):
        return len(self.init_loc)
