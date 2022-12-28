import torch
import pytorch_lightning as pl


from .dataset import NBodyDataset, QM9Dataset
from .model import GNN, EquivariantGNN


DATASETS = {
    "nbody": NBodyDataset,
    "qm9": QM9Dataset
}

MODELS = {
    "gnn": GNN,
    "egnn": EquivariantGNN
}

CRITERIONS = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
}


class GNNModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.model = MODELS[self._config["model_type"]](**self._config["model_params"])
        criterion_type = "mse" if "criterion_type" not in config else config["criterion_type"]
        self.criterion = CRITERIONS[criterion_type]()

    def train_dataset(self):
        params = {} if "dataset_params" not in self._config else self._config["dataset_params"]
        dataset = DATASETS[self._config["dataset_type"]](split="train", **params)
        return dataset

    def train_dataloader(self):
        dataset = self.train_dataset()
        params = {} if "loader_params" not in self._config else self._config["loader_params"]
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, **params)
        return loader

    def val_dataset(self):
        params = {} if "dataset_params" not in self._config else self._config["dataset_params"]
        dataset = DATASETS[self._config["dataset_type"]](split="valid", **params)
        return dataset

    def val_dataloader(self):
        dataset = self.val_dataset()
        params = {} if "loader_params" not in self._config else self._config["loader_params"]
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, **params)
        return loader

    def test_dataset(self):
        params = {} if "dataset_params" not in self._config else self._config["dataset_params"]
        dataset = DATASETS[self._config["dataset_type"]](split="test", **params)
        return dataset

    def test_dataloader(self):
        dataset = self.test_dataset()
        params = {} if "loader_params" not in self._config else self._config["loader_params"]
        loader = torch.utils.data.DataLoader(dataset, shuffle=False, **params)
        return loader

    def configure_optimizers(self):
        params = {} if "optimizer_params" not in self._config else self._config["optimizer_params"]
        optimizer = torch.optim.Adam(self.parameters(), **params)
        return optimizer

    def training_step(self, batch, batch_idx, log_key="loss"):
        batch = prepare_batch(batch)
        graph, labels = batch
        preds, _, _ = self.model(*graph, batch_size=len(labels))
        loss = self.criterion(preds, labels)
        self.log(log_key, loss, batch_size=len(labels))
        return {log_key: loss}

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, log_key="test_loss")

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, log_key="val_loss")


def prepare_batch(batch):
    graphs, labels = batch
    labels = torch.vstack(list(labels))

    nodes, edge_ids, edges = graphs
    for i in range(len(nodes) - 1):
        graph_size = nodes[i].size()[0]
        edge_ids[0][i + 1:] += graph_size
        edge_ids[1][i + 1:] += graph_size

    nodes = torch.vstack(list(nodes))
    edges = torch.vstack(list(edges))
    edge_ids[0] = torch.hstack(list(edge_ids[0]))
    edge_ids[1] = torch.hstack(list(edge_ids[1]))

    return (nodes.float(), edge_ids, edges.float()), labels.float()
