#! /usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

from shape_completion_training.model.model_runner import ModelRunner
from ycb_video_pytools import ycb_video_dataset


def train_main(args, seed):
    dataset = ycb_video_dataset.YCBReconstructionDataset(args.dataset_dir)
    tf_train_dataset = dataset.load(mode='train')
    tf_validation_dataset = dataset.load(mode='val')

    model_params = json.load(open(args.model_params, 'r'))
    net = ModelRunner(model_params,
                      training=True,
                      rootdir='trials',
                      trial_name=args.trial_name,
                      batch_size=args.batch_size,
                      validation_size=args.validation_size,
                      validation_summary_every=args.summary_every,
                      train_summary_every=args.summary_every,
                      validation_on_end_of_epoch=True,
                      )
    net.train(tf_train_dataset, tf_validation_dataset)


def eval_main(args, seed):
    dataset = ycb_video_dataset.YCBReconstructionDataset(args.dataset_dir)
    tf_test_dataset = dataset.load(mode='test')

    net = ModelRunner(params=None, training=False, rootdir='trials')
    net.build_model(tf_test_dataset)
    net.evaluate(tf_test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_subparser = subparsers.add_parser('train')
    train_subparser.add_argument('dataset_dir', help='dataset directory', type=pathlib.Path, nargs='+')
    train_subparser.add_argument('model_params', type=pathlib.Path)
    train_subparser.add_argument('--seed', type=int)
    train_subparser.add_argument('--summary-every', type=int, default=500)
    train_subparser.add_argument('--validation-size', type=int, default=100)
    train_subparser.add_argument('-l', '--trial-name', default='nickname')
    train_subparser.add_argument('--batch-size', type=int, default=16)
    train_subparser.set_defaults(func=train_main)

    test_subparser = subparsers.add_parser('test')
    test_subparser.add_argument('dataset_dir', help='dataset directory', type=pathlib.Path, nargs='+')
    test_subparser.add_argument('--seed', type=int)
    test_subparser.set_defaults(func=eval_main)

    args = parser.parse_args()

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
