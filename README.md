# EGNN

Simple and clean implementation of Graph Neural Networks and E(n)
Equivariant Graph Neural Networks from the [paper](https://arxiv.org/pdf/2102.09844v3.pdf).

## Quick Start

Install the package
```bash
git clone git@github.com:stdereka/egnn.git
cd egnn
pip install -e .
```

Download [NBody and QM9](https://drive.google.com/drive/folders/19AUFCMlDXVdOENtz2HRNHoR13i8OX7Ej?usp=share_link) datasets and unpack them.

Then you can run `egnn` package as a Python 3 module. **Note**: check
dataset root directory in `.yaml` config. 
```bash
python -m egnn -c config/qm9_egnn_cv.yaml
```
This command trains EGNN model on QM9 dataset and stores Tensorboard
logs in `./logs`. You may find other config examples in `./config`.

## Using as a Library
For more details see `help()` for `GNN` and `EGNN` classes.
```python
import torch
from egnn import GNN

gnn = GNN(
    input_node_dim=3,
    input_edge_dim=2,
    output_dim=1,
    hidden_dim=64,
    num_layers=3,
)

# 3 nodes with 3 features
node_features = torch.tensor(
    [[0.1, 0.2, 0.3],
     [23.0, 0.0, 1.0],
     [0.0, 0.0, 10.1]]
)

# 2 edges: 0-1 and 1-2
edge_ids = [
    torch.tensor([0, 1]),
    torch.tensor([1, 2]),
]

# Each edge has 2 features
edge_features = torch.tensor(
    [[1.1, 0.2],
     [2.0, 0.0]],
)

out = gnn(node_features, edge_ids, edge_features, 1)[0]
# Model prediction for each node
# tensor([[-0.0889],
#         [-4.1104],
#         [ 1.2860]], grad_fn=<AddmmBackward0>)
```
