from argparse import ArgumentParser

import yaml
import pytorch_lightning as pl
from .lightning_module import GNNModule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    module = GNNModule(config)
    logger = pl.loggers.TensorBoardLogger("./logs", name=args.config)
    trainer = pl.Trainer(
        logger=logger,
        gpus=-1,
        **config["trainer_params"]
    )

    trainer.fit(module)
    trainer.test()
