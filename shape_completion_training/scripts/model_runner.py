#!/usr/bin/env python
import argparse
import json
import logging
import pathlib
from copy import deepcopy

import numpy as np
import tensorflow as tf

from moonshine import experiments_util
from moonshine.loss_utils import sigmoid_cross_entropy_with_logits
from moonshine.tensorflow_train_test_loop import evaluate, train
from shape_completion_training.model import network
from shape_completion_training.model.nn_tools import make_metrics_function
from ycb_video_pytools.ycb_video_dataset import YCBReconstructionDataset


def train_func(args, seed: int):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    # Model parameters
    model_hparams = json.load(open(args.model_hparams, 'r'))

    # Datasets
    dataset = YCBReconstructionDataset(args.dataset_dirs)
    train_tf_dataset = dataset.load(mode='train', take=args.take)
    val_tf_dataset = dataset.load(mode='val', take=args.take)
    train_tf_dataset = train_tf_dataset.shuffle(seed=args.seed, buffer_size=1024).batch(args.batch_size, drop_remainder=True)
    val_tf_dataset = val_tf_dataset.batch(args.batch_size, drop_remainder=True)

    # Copy parameters of the dataset into the model
    model_hparams['dynamics_dataset_hparams'] = dataset.hparams
    model_hparams['batch_size'] = args.batch_size
    model = network.get_model(model_hparams['network'])
    net = model(params=deepcopy(model_hparams), batch_size=args.batch_size)

    ###############
    # Train
    ###############
    train(keras_model=net,
          model_hparams=model_hparams,
          train_tf_dataset=train_tf_dataset,
          val_tf_dataset=val_tf_dataset,
          dataset_dirs=args.dataset_dirs,
          seed=seed,
          batch_size=args.batch_size,
          epochs=model_hparams['epochs'],
          loss_function=sigmoid_cross_entropy_with_logits,
          metrics_function=make_metrics_function(sigmoid_cross_entropy_with_logits),
          checkpoint=args.checkpoint,
          log_path=log_path,
          log_scalars_every=args.log_scalars_every)


def eval_func(args, seed: int):
    tf.config.experimental_run_functions_eagerly(True)

    ###############
    # Dataset
    ###############
    test_dataset = YCBReconstructionDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.load(mode=args.mode, take=args.take, shard=args.shard)

    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)

    ###############
    # Model
    ###############
    model_hparams_file = args.checkpoint / 'hparams.json'
    model_hparams = json.load(open(model_hparams_file, 'r'))

    model = network.get_model(model_hparams['network'])
    net = model(params=model_hparams, batch_size=args.batch_size)

    ###############
    # Evaluate
    ###############
    evaluate(keras_model=net,
             test_tf_dataset=test_tf_dataset,
             loss_function=sigmoid_cross_entropy_with_logits,
             metrics_function=make_metrics_function(sigmoid_cross_entropy_with_logits),
             checkpoint_path=args.checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=16)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--take', type=int, help='take a subset of the dataset')
    train_parser.add_argument('--shard', type=int, help='take every nth element of the dataset')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches', default=500)
    train_parser.set_defaults(func=train_func)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--take', type=int, help='take a subset of the dataset')
    eval_parser.add_argument('--batch-size', type=int, default=16)
    eval_parser.add_argument('--mode', type=str, choices=['val', 'train'], default='val')
    eval_parser.add_argument('--shard', type=int, help='take every nth element of the dataset')
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval_func)

    args = parser.parse_args()
    tf.get_logger().setLevel(logging.ERROR)

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Random seed: {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
