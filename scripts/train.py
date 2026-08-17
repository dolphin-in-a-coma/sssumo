"""Command-line entrypoint for training, for unattended runs on a remote VM.

The loop itself lives in `sssumo.training.train`, which notebooks/Train.ipynb
also calls -- this file only parses arguments and sets up wandb.

Directory layout expected under --root-dir (Config derives these):
    <root>/data/     the *_tangential_velocity_data.csv files
    <root>/weights/  checkpoints, written as <experiment>_<epoch>.pth
    <root>/logs/     text log, written as <experiment>.txt

Example:
    python scripts/train.py --config configs/config-0423-ModGaussian_ampl.yaml \
        --root-dir /content/sssumo
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')  # headless: no display on the VM

import wandb  # noqa: E402

from sssumo.training import train  # noqa: E402
from sssumo.utils import Config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='path to a YAML config in configs/')
    parser.add_argument('--root-dir', required=True,
                        help='root holding data/, weights/ and logs/')
    parser.add_argument('--experiment-name', default=None,
                        help='override the run name (defaults to the config filename); '
                             'checkpoints and logs are named after it, so change it to '
                             'avoid overwriting an earlier run')
    parser.add_argument('--resume', action='store_true',
                        help='continue from the highest epoch checkpoint already in weights/, '
                             'ignoring start_with_weights in the config')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='override optimizer steps per epoch (smoke tests)')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='override the number of epochs (smoke tests)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='override trials per step')
    parser.add_argument('--organic-eval-every', type=int, default=5,
                        help='epochs between organic test evaluations; 0 disables')
    parser.add_argument('--eval-datapoints', type=int, default=None,
                        help='cap trials per organic evaluation (smoke tests)')
    parser.add_argument('--wandb-project', default='submovement_detector',
                        help='wandb project to log to')
    parser.add_argument('--wandb-key-file', default=None,
                        help='path to a file containing only the wandb API key; falls back to '
                             '$WANDB_KEY. Never pass the key itself -- it would land in argv.')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
    return parser.parse_args()


def start_wandb(config, args):
    """Log in without the key ever reaching argv, stdout or the config dump."""
    if args.wandb_key_file:
        with open(args.wandb_key_file) as f:
            key = f.read().strip()
    else:
        key = os.getenv('WANDB_KEY')

    wandb.login(key=key)
    del key
    wandb.init(project=args.wandb_project, name=config.experiment_name,
               config=config.to_dict(), save_code=True)


def main():
    args = parse_args()

    config = Config(args.config, root_dir=args.root_dir)
    config.experiment_name = (args.experiment_name
                              or os.path.basename(args.config).replace('.yaml', ''))

    if args.num_samples is not None:
        config.num_samples = args.num_samples
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    use_wandb = not args.no_wandb
    if use_wandb:
        start_wandb(config, args)

    try:
        train(config,
              organic_eval_every=args.organic_eval_every,
              eval_datapoints=args.eval_datapoints,
              resume=args.resume,
              plot=False)
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
