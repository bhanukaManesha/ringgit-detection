import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy

from tensorboard.plugins.hparams import api as hp

from common import *

class YOLOMetrics:

    class EarlyStoppingCallback(keras.callbacks.EarlyStopping):
        def __init__(self):
            super().__init__(monitor='val_loss', mode='min', verbose=1, patience=50)

    class TensorboardCallback(keras.callbacks.TensorBoard):
        def __init__(self, logdir):
            # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            super().__init__(log_dir = logdir, histogram_freq=1, profile_batch=0)

    class HistoryCheckpointCallback(keras.callbacks.Callback):
        def __init__(self, folder):
            self.folder = folder
        def on_train_begin(self, logs={}):
            with open('{}/model.json'.format(self.folder), 'w') as f:
                json.dump(json.loads(self.model.to_json()), f)
            with open('{}/history.txt'.format(self.folder), 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        def on_epoch_end(self, epoch, logs={}):
            keys = ['loss','P_', 'XY_','C_']
            h = ' - '. join(['{}: {:.4f}'.format(k, logs[k]) for k in keys])
            h = h + ' // ' + ' - '. join(['val_{}: {:.4f}'.format(k, logs['val_'+k]) for k in keys])
            h = '{:03d} : '.format(epoch) + h
            with open('{}/history.txt'.format(self.folder), 'a') as f:
                f.write(h + '\n')

    @staticmethod
    def PR_(fact,pred):
        fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        # Truth
        fact_conf = fact[:,:,0]
        fw = fact[:,:,3] * WIDTH
        fh = fact[:,:,4] * HEIGHT
        fx = fact[:,:,0] * GRID_WIDTH - fw/2
        fy = fact[:,:,1] * GRID_HEIGHT - fh/2

        # Prediction
        pred_conf = fact[:,:,0]
        pw = pred[:,:,3] * WIDTH
        ph = pred[:,:,4] * HEIGHT
        px = pred[:,:,0] * GRID_WIDTH - pw/2
        py = pred[:,:,1] * GRID_HEIGHT - ph/2

        # IOU
        intersect = (tf.minimum(fx+fw, px+pw) - tf.maximum(fx, px)) * (tf.minimum(fy+fh, py+ph) - tf.maximum(fy, py))
        union = (fw * fh) + (pw * ph) - intersect
        iou =  (intersect/union)

        detection = tf.cast(pred_conf >= DETECTION_PARAMETER, dtype=tf.float32)
        no_detection = tf.cast(pred_conf < DETECTION_PARAMETER, dtype=tf.float32)

        iou_greater = tf.cast(iou >= DETECTION_PARAMETER, dtype=tf.float32)
        iou_less = tf.cast(iou < DETECTION_PARAMETER, dtype=tf.float32)

        tp = tf.math.count_nonzero(detection * iou_greater, dtype=tf.float32, axis=1)
        fp = tf.math.count_nonzero(detection * iou_less, dtype=tf.float32, axis=1)

        fn = tf.math.count_nonzero(no_detection * fact_conf, dtype=tf.float32, axis=1)

        precision = (tp / (tp + fp))

        return K.switch(
            tf.equal(tf.math.reduce_mean(precision), 0),
            0.0,
            tf.math.reduce_mean(precision)
            )

    @staticmethod
    def RC_(fact,pred):

        fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        # Truth
        fact_conf = fact[:,:,0]
        fw = fact[:,:,3] * WIDTH
        fh = fact[:,:,4] * HEIGHT
        fx = fact[:,:,0] * GRID_WIDTH - fw/2
        fy = fact[:,:,1] * GRID_HEIGHT - fh/2

        # Prediction
        pred_conf = fact[:,:,0]
        pw = pred[:,:,3] * WIDTH
        ph = pred[:,:,4] * HEIGHT
        px = pred[:,:,0] * GRID_WIDTH - pw/2
        py = pred[:,:,1] * GRID_HEIGHT - ph/2

        # IOU
        intersect = (tf.minimum(fx+fw, px+pw) - tf.maximum(fx, px)) * (tf.minimum(fy+fh, py+ph) - tf.maximum(fy, py))
        union = (fw * fh) + (pw * ph) - intersect
        iou =  (intersect/union)

        detection = tf.cast(pred_conf >= DETECTION_PARAMETER, dtype=tf.float32)
        no_detection = tf.cast(pred_conf < DETECTION_PARAMETER, dtype=tf.float32)

        iou_greater = tf.cast(iou >= DETECTION_PARAMETER, dtype=tf.float32)
        iou_less = tf.cast(iou < DETECTION_PARAMETER, dtype=tf.float32)

        tp = tf.math.count_nonzero(detection * iou_greater, dtype=tf.float32, axis=1)

        fp = tf.math.count_nonzero(detection * iou_less, dtype=tf.float32, axis=1)

        fn = tf.math.count_nonzero(no_detection * fact_conf, dtype=tf.float32, axis=1)

        total = tf.math.count_nonzero(fact_conf, dtype=tf.float32, axis=1)

        recall = tp / total

        return switch(
            tf.equal(tf.math.reduce_mean(recall), 0),
            0.0,
            tf.math.reduce_mean(recall)
            )
    @staticmethod
    def XY_(fact, pred):
        fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        # Truth
        fact_conf = fact[:,:,0]
        fw = fact[:,:,3] * GRID_WIDTH
        fh = fact[:,:,4] * GRID_HEIGHT
        fx = fact[:,:,0] * GRID_WIDTH - fw/2
        fy = fact[:,:,1] * GRID_HEIGHT - fh/2
        # Prediction
        pw = pred[:,:,3] * GRID_WIDTH
        ph = pred[:,:,4] * GRID_HEIGHT
        px = pred[:,:,0] * GRID_WIDTH - pw/2
        py = pred[:,:,1] * GRID_HEIGHT - ph/2
        # IOU
        intersect = (K.minimum(fx+fw, px+pw) - K.maximum(fx, px)) * (K.minimum(fy+fh, py+ph) - K.maximum(fy, py))
        union = (fw * fh) + (pw * ph) - intersect
        nonzero_count = tf.math.count_nonzero(fact_conf, dtype=tf.float32, axis=1)
        sum_per_row = K.sum((intersect / union) * fact_conf, axis=1) / nonzero_count
        mean = tf.math.reduce_mean(sum_per_row)
        # Mean.
        return K.switch(tf.equal(mean, 0), 0.0, mean)

    @staticmethod
    def C_(fact, pred):
        fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        # Truth
        fact_conf = fact[:,:,0]
        fact_cat = fact[:,:,5:]
        # Prediction
        pred_cat = pred[:,:,5:]
        # CLASSIFICATION
        nonzero_count = tf.math.count_nonzero(fact_conf, dtype=tf.float32)
        return K.switch(
            tf.equal(nonzero_count, 0),
            1.0,
            K.sum(categorical_accuracy(fact_cat, pred_cat) * fact_conf) / nonzero_count
        )

    @staticmethod
    def P_(fact, pred):
        fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
        # Truth
        fact_conf = fact[:,:,0]
        # Prediction
        pred_conf = pred[:,:,0]
        # PROBABILITY
        return binary_accuracy(fact_conf, pred_conf)