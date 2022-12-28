import os

import torch
import numpy as np


class QM9Dataset(torch.utils.data.Dataset):
    NUM_UNIQUE_ATOMS = 5
    PROPERTIES = 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',\
        'omega1', 'zpve_thermo', 'U0_thermo', 'U_thermo', 'H_thermo', 'G_thermo', 'Cv_thermo'

    def __init__(self, root, split="train", label="Cv"):
        data = np.load(os.path.join(root, f"{split}.npz"))

        if label not in self.PROPERTIES:
            raise ValueError(f"Unknown label. Should be in {self.PROPERTIES}")

        labels = data[label]
        coords = data["positions"]
        charges = data["charges"]

        one_hots = (np.arange(charges.max() + 1) == charges[..., None]).astype(np.float64)[:, :, np.unique(charges)]

        first = []
        second = []
        for i in range(coords.shape[1]):
            for j in range(coords.shape[1]):
                if i != j:
                    first.append(i)
                    second.append(j)

        edges = [torch.tensor(first), torch.tensor(second)]
        self.edges = edges
        self.one_hots = torch.tensor(one_hots)
        self.labels = torch.tensor(labels)
        self.coords = torch.tensor(coords)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = torch.hstack([self.coords[item], self.one_hots[item]]), self.edges, torch.ones((len(self.edges[0]), 1))
        label = self.labels[item]
        return data, label
