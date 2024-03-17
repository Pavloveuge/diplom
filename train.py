import argparse
from hyperpyyaml import load_hyperpyyaml
from transformers import get_scheduler
from train_loop import train_loop
from torch.utils.data import DataLoader


def train(config):
    model
    optimizer = config.optimizer(model.parameters())
    lr_scheduler = get_scheduler(name=config.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=config.warmup_steps)

    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=config.collate_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My example explanation')
    parser.add_argument(
        '--path_to_config',
        type=str,
        help='path to train config'
    )
    args = parser.parse_args()

    config = load_hyperpyyaml(args.path_to_config)

    train(config)
