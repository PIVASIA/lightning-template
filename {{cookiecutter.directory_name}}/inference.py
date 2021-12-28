import os
import numpy as np
import argparse

import pytorch_lightning as pl

from models.model import {{cookiecutter.model_name}}, {{cookiecutter.model_task}}
from datasets.datamodule import {{cookiecutter.datamodule_name}}


def _parse_args():
    parser = argparse.ArgumentParser(description="{{cookiecutter.project_description}}")
    # model parameters
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to testing data')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Path to output dir')
    parser.add_argument('--conf-t', type=float, default=0.0, 
                        help='confidence thershold')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--gpu-ids', default=0, type=int, nargs="+",
                        help='use which gpu to train (default=0)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    assert args.model_path is not None

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    dm = {{cookiecutter.datamodule_name}}(
                test_path=args.test_path,
                test_batch_size=args.test_batch_size,
                seed=args.seed)
    dm.setup("predict")

    model = {{cookiecutter.model_name}}.load_from_checkpoint(args.model_path)
    task = {{cookiecutter.model_task}}(model, args.output_dir, threshold=args.conf_t)

    trainer = pl.Trainer(gpus=args.gpu_ids)
    trainer.predict(task, datamodule=dm)

if __name__ == "__main__":
    main()