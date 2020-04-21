#! /usr/bin/env python
import argparse
import numpy as np
import pathlib

import tensorflow as tf

from shape_completion_training.model.network import Network
from ycb_video_pytools import ycb_video_dataset

params = {
    'num_latent_layers': 200,
    'translation_pixel_range_x': 10,
    'translation_pixel_range_y': 10,
    'translation_pixel_range_z': 10,
    'is_u_connected': False,
    'final_activation': 'None',
    'unet_dropout_rate': 0.5,
    'use_final_unet_layer': False,
    'simulate_partial_completion': False,
    'simulate_random_partial_completion': False,
    # 'network': 'VoxelCNN',
    # 'network': 'VAE_GAN',
    'network': 'AutoEncoder',
    # 'network': 'Conditional_VCNN',
    'stacknet_version': 'v2',
    'turn_on_prob': 0.00000,
    'turn_off_prob': 0.0,
    'loss': 'cross_entropy',
    'multistep_loss': True,
    'flooding_level': None,
}


def train_main(args, seed):
    dataset = ycb_video_dataset.YCBReconstructionDataset(args.dataset_dir)
    tf_train_dataset = dataset.load(mode='train').take(128)
    tf_validation_dataset = dataset.load(mode='val').take(128)

    net = Network(params, training=True, rootdir='trials')
    net.train(tf_train_dataset, tf_validation_dataset)


def eval_main(args, seed):
    dataset = ycb_video_dataset.YCBReconstructionDataset(args.dataset_dir)
    tf_test_dataset = dataset.load(mode='test')

    net = Network(params=None, training=False, rootdir='trials')
    net.build_model(tf_test_dataset)
    net.evaluate(tf_test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_subparser = subparsers.add_parser('train')
    train_subparser.add_argument('dataset_dir', help='dataset directory', type=pathlib.Path, nargs='+')
    train_subparser.add_argument('--seed', type=int)
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
