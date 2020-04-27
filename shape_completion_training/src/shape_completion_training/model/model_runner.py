import os
import time

import progressbar
import tensorflow as tf

from shape_completion_training.model import filepath_tools
from shape_completion_training.model.augmented_ae import Augmented_VAE
from shape_completion_training.model.auto_encoder import AutoEncoder
from shape_completion_training.model.conditional_vcnn import ConditionalVCNN
from shape_completion_training.model.vae import VAE, VAE_GAN
from shape_completion_training.model.voxelcnn import VoxelCNN


def get_model(network_type):
    if network_type == 'VoxelCNN':
        return VoxelCNN
    elif network_type == 'AutoEncoder':
        return AutoEncoder
    elif network_type == 'VAE':
        return VAE
    elif network_type == 'VAE_GAN':
        return VAE_GAN
    elif network_type == 'Augmented_VAE':
        return Augmented_VAE
    elif network_type == 'Conditional_VCNN':
        return ConditionalVCNN
    else:
        raise Exception('Unknown Model Type')


def instantiate_model(params, batch_size):
    if params['network'] == 'VoxelCNN':
        model = VoxelCNN(params, batch_size=batch_size)
    elif params['network'] == 'AutoEncoder':
        model = AutoEncoder(params, batch_size=batch_size)
    elif params['network'] == 'VAE':
        model = VAE(params, batch_size=batch_size)
    elif params['network'] == 'VAE_GAN':
        model = VAE_GAN(params, batch_size=batch_size)
    elif params['network'] == 'Augmented_VAE':
        model = Augmented_VAE(params, batch_size=batch_size)
    elif params['network'] == 'Conditional_VCNN':
        model = ConditionalVCNN(params, batch_size=batch_size)
    else:
        raise Exception('Unknown Model Type')
    return model


class ModelRunner:
    def __init__(self,
                 params=None,
                 trial_name=None,
                 training=False,
                 rootdir='trials',
                 train_summary_every: int = 100,
                 validation_summary_every: int = 100,
                 validation_size: int = 100,
                 validation_on_end_of_epoch: bool = True,
                 batch_size: int = 16):
        self.train_summary_every = train_summary_every
        self.validation_on_end_of_epoch = validation_on_end_of_epoch
        self.batch_size = batch_size
        self.validation_summary_every = validation_summary_every
        self.validation_size = validation_size
        if not training:
            self.batch_size = 1
        self.side_length = 64
        self.num_voxels = self.side_length ** 3

        file_fp = os.path.dirname(__file__)
        fp = filepath_tools.get_trial_directory(rootdir, expect_reuse=(params is None), nick=trial_name)
        self.trial_name = fp.split('/')[-1]
        self.params = filepath_tools.handle_params(file_fp, fp, params)

        self.trial_fp = fp
        self.checkpoint_path = os.path.join(fp, "training_checkpoints/")

        train_log_dir = os.path.join(fp, 'logs/train')
        test_log_dir = os.path.join(fp, 'logs/test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.model = instantiate_model(params=self.params, batch_size=self.batch_size)

        self.num_batches = None

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        epoch=tf.Variable(0),
                                        train_time=tf.Variable(0.0),
                                        optimizer=self.model.opt,
                                        net=self.model.get_model())
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1)
        self.restore()

    def restore(self):
        status = self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()

        # Suppress warning 
        if self.manager.latest_checkpoint:
            status.assert_existing_objects_matched()

    def count_params(self):
        self.model.summary()

    def build_model(self, dataset):
        elem = dataset.take(self.batch_size).batch(self.batch_size)
        tf.summary.trace_on(graph=True, profiler=False)
        self.model.predict(elem)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.ckpt.step.numpy())

        # tf.keras.utils.plot_model(self.model.get_model(), os.path.join(self.trial_fp, 'network.png'),
        #                           show_shapes=True)

    def write_summary(self, writer, summary_dict):
        with writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())

    def train_batch(self, train_dataset, val_dataset):
        self.num_batches = 0
        t0 = time.time()
        validation_iterator = iter(val_dataset.repeat())
        for batch in progressbar.progressbar(train_dataset):
            self.num_batches += 1
            self.ckpt.step.assign_add(1)

            metrics = self.model.train_step(batch)
            if self.ckpt.step.numpy() % self.train_summary_every == 0:
                self.write_summary(self.train_summary_writer, metrics)

            if self.ckpt.step.numpy() % self.validation_summary_every == 0:
                summaries = {}
                for i in range(self.validation_size):
                    batch = next(validation_iterator)
                    metrics = self.model.val_step(batch)
                    for k, v in metrics.items():
                        if k not in summaries:
                            summaries[k] = []
                        summaries[k].append(v)
                mean_summary_dict = dict([(k, tf.math.reduce_mean(v)) for (k, v) in summaries.items()])
                self.write_summary(self.test_summary_writer, mean_summary_dict)

            self.ckpt.train_time.assign_add(time.time() - t0)

        # End of epoch
        if self.validation_on_end_of_epoch:
            self.validation(progressbar.progressbar(val_dataset))

        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def validation(self, iterable):
        summaries = {}
        for batch in iterable:
            metrics = self.model.val_step(batch)
            for k, v in metrics.items():
                if k not in summaries:
                    summaries[k] = []
                summaries[k].append(v)
        mean_summary_dict = dict([(k, tf.math.reduce_mean(v)) for (k, v) in summaries.items()])
        self.write_summary(self.test_summary_writer, mean_summary_dict)

    def train(self, train_dataset, validation_dataset):
        self.build_model(train_dataset)
        self.count_params()

        batched_train_dataset = train_dataset.batch(self.batch_size, drop_remainder=True).prefetch(64)
        if validation_dataset is not None:
            batched_validation_dataset = validation_dataset.batch(self.batch_size, drop_remainder=True).prefetch(64)
        else:
            batched_validation_dataset = None

        num_epochs = 1000
        while self.ckpt.epoch < num_epochs:
            self.ckpt.epoch.assign_add(1)
            print('')
            print('==  Epoch {}/{}  '.format(self.ckpt.epoch.numpy(), num_epochs) + '=' * 25 \
                  + ' ' + self.trial_name + ' ' + '=' * 20)
            self.train_batch(batched_train_dataset, batched_validation_dataset)
            print('=' * 48)

    def train_and_test(self, dataset):
        train_ds = dataset

        self.train(train_ds, None)
        self.count_params()

    def evaluate(self, dataset):
        self.validation(dataset.batch(self.batch_size, drop_remainder=True))
