import datetime
import os
import time

from tensorboard.plugins.mesh import summary as mesh_summary
# from tensorboard.plugins.mesh import summary_v2 as mesh_summary

import progressbar
import tensorflow as tf

from shape_completion_training.model import filepath_tools
from shape_completion_training.model.auto_encoder import AutoEncoder
from shape_completion_training.model.augmented_ae import Augmented_VAE
from shape_completion_training.model.voxelcnn import VoxelCNN
from shape_completion_training.model.vae import VAE, VAE_GAN
from shape_completion_training.model.conditional_vcnn import ConditionalVCNN


class Network:
    def __init__(self, params=None, trial_name=None, training=False, rootdir='trials'):
        self.batch_size = 16
        if not training:
            self.batch_size = 1
        self.side_length = 64
        self.num_voxels = self.side_length ** 3

        file_fp = os.path.dirname(__file__)
        fp = filepath_tools.get_trial_directory(rootdir,
                                                expect_reuse=(params is None),
                                                nick=trial_name)
        self.trial_name = fp.split('/')[-1]
        self.params = filepath_tools.handle_params(file_fp, fp, params)

        self.trial_fp = fp
        self.checkpoint_path = os.path.join(fp, "training_checkpoints/")

        train_log_dir = os.path.join(fp, 'logs/train')
        test_log_dir = os.path.join(fp, 'logs/test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        if self.params['network'] == 'VoxelCNN':
            self.model = VoxelCNN(self.params, batch_size=self.batch_size)
        # if self.params['network'] == 'StackedVoxelCNN':
        #     self.model = StackedVoxelCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'AutoEncoder':
            self.model = AutoEncoder(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'VAE':
            self.model = VAE(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'VAE_GAN':
            self.model = VAE_GAN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'Augmented_VAE':
            self.model = Augmented_VAE(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'Conditional_VCNN':
            self.model = ConditionalVCNN(self.params, batch_size=self.batch_size)
        else:
            raise Exception('Unknown Model Type')

        self.num_batches = None

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        epoch=tf.Variable(0),
                                        train_time=tf.Variable(0.0),
                                        optimizer=self.model.opt, net=self.model.get_model())
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1)
        self.restore()

    def restore(self):
        status = self.ckpt.restore(self.manager.latest_checkpoint)

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

    def write_summary(self, summary_dict, batch, output):
        with self.train_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())

                gt_occ_point_cloud = tf.where(tf.squeeze(batch['gt_occ'][0]))
                predicted_occ_point_cloud = tf.where(tf.squeeze(output['predicted_occ'][0]))
                red = [1.0, 0.0, 0.0]
                mesh_summary.op('predicted_occ',
                                vertices=predicted_occ_point_cloud,
                                colors=tf.ones_like(predicted_occ_point_cloud) * red)
                mesh_summary.op('gt_occ',
                                vertices=gt_occ_point_cloud,
                                colors=tf.ones_like(gt_occ_point_cloud) * red)

    def train_batch(self, dataset):
        if self.num_batches is not None:
            max_size = str(self.num_batches)
        else:
            max_size = '???'

        self.num_batches = 0
        t0 = time.time()
        for batch in progressbar.progressbar(dataset):
            self.num_batches += 1
            self.ckpt.step.assign_add(1)

            summary_dict, output = self.model.train_step(batch)
            time_str = str(datetime.timedelta(seconds=int(self.ckpt.train_time.numpy())))
            self.write_summary(summary_dict, batch, output)
            self.ckpt.train_time.assign_add(time.time() - t0)
            t0 = time.time()

        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        print("loss {:1.3f}".format(summary_dict['loss'].numpy()))

    def train(self, dataset):
        self.build_model(dataset)
        self.count_params()
        # dataset = dataset.shuffle(10000)

        # batched_ds = dataset.batch(self.batch_size, drop_remainder=True).prefetch(64)
        batched_ds = dataset.batch(self.batch_size).prefetch(64)

        num_epochs = 1000
        while self.ckpt.epoch < num_epochs:
            self.ckpt.epoch.assign_add(1)
            print('')
            print('==  Epoch {}/{}  '.format(self.ckpt.epoch.numpy(), num_epochs) + '=' * 25 \
                  + ' ' + self.trial_name + ' ' + '=' * 20)
            self.train_batch(batched_ds)
            print('=' * 48)

    def train_and_test(self, dataset):
        train_ds = dataset

        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))
