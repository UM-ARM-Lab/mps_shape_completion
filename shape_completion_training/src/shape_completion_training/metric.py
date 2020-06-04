import tensorflow as tf


class Metric:

    @staticmethod
    def is_better_than(a, b):
        raise NotImplementedError()

    @staticmethod
    def key():
        raise NotImplementedError()

    @staticmethod
    def worst():
        raise NotImplementedError()


class LossMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        return a < b

    @staticmethod
    def key():
        return "loss"

    @staticmethod
    def worst():
        return 1000


class AccuracyMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        if b is None:
            return True
        return a > b

    @staticmethod
    def key():
        return "accuracy"

    @staticmethod
    def worst():
        return 0


def recall(y_true, y_pred, threshold=0.5):
    true_positives = tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred > threshold, tf.float32), ), tf.float32)
    false_negatives = tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred <= threshold, tf.float32)), tf.float32)
    return tf.math.divide_no_nan(true_positives, true_positives + false_negatives)


def precision(y_true, y_pred, threshold=0.5):
    true_positives = tf.cast(tf.math.count_nonzero(y_true * tf.cast(y_pred > threshold, tf.float32)), tf.float32)
    false_positives = tf.cast(tf.math.count_nonzero((1 - y_true) * tf.cast(y_pred <= threshold, tf.float32)), tf.float32)
    return tf.math.divide_no_nan(true_positives, true_positives + false_positives)
