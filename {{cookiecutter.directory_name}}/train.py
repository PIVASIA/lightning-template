import numpy as np
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

import mlflow
import mlflow.pytorch

from models.model import {{cookiecutter.model_name}}
from datasets.datamodule import {{cookiecutter.datamodule_name}}
MLFLOW_TRACKING_URI = "databricks"
MLFLOW_EXPERIMENT = "/Users/hunglv@piv.asia/{{cookiecutter.project_name}}"


def _parse_args():
    parser = argparse.ArgumentParser(description="{{cookiecutter.project_description}}")
    # model parameters
    parser.add_argument('--train-path', type=str, default=None,
                        help='Path to training data')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to testing data')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train-batch-size', type=int, default=32,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--gpu-ids', default=0, type=int, nargs="+",
                        help='use which gpu to train (default=0)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # name
    parser.add_argument('--resume', type=str, default=None, 
                        help='resume training')

    args = parser.parse_args()
    return args


def handle_train(args):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    
    # TODO: ADD PARAMETERS AS MANY AS YOU NEED
    dm = {{cookiecutter.datamodule_name}}(
                    train_path=args.train_path,
                    test_path=args.test_path,
                    train_batch_size=args.train_batch_size,
                    test_batch_size=args.test_batch_size,
                    seed=args.seed)
    dm.setup(stage="fit")

    # TODO: ADD PARAMETERS AS MANY AS YOU NEED
    model = {{cookiecutter.model_name}}(
                    n_epochs=args.epochs, 
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # monitored quantity
        filename='Sequence-{epoch:02d}-{val_loss:.5f}',
        save_top_k=3, # save the top 3 models
        mode='min', # mode of the monitored quantity for optimization
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, patience=3, 
        verbose=False, 
        mode="max"
    )
    callbacks=[checkpoint_callback, early_stop_callback]

    mlflow.pytorch.autolog()
    # use float (e.g 1.0) to set val frequency in epoch
    # if val_check_interval is integer, val frequency is in batch step
    training_params = {
        "callbacks": callbacks,
        "gpus": args.gpu_ids,
        "val_check_interval": 1.0,
        "max_epochs": args.epochs
    }
    if args.resume is not None:
        training_params["resume_from_checkpoint"] = args.resume
    
    trainer = pl.Trainer(**training_params)
    trainer.fit(model, dm)


def main():
    args = _parse_args()
    handle_train(args)


if __name__ == "__main__":
    main()