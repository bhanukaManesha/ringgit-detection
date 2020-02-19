#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

from common import *
from utils import *

from datetime import datetime
import os
import shutil
import math
import numpy as np
import cv2
import json
import glob

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow import keras
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from tensorflow.keras.backend import *

from generator import generator



os.environ["CUDA_VISIBLE_DEVICES"]="0"

def loss(fact, pred):
    fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])

    # Truth
    fact_conf = fact[:,:,0]
    fact_x    = fact[:,:,1]
    fact_y    = fact[:,:,2]
    fact_w    = fact[:,:,3]
    fact_h    = fact[:,:,4]
    fact_cat  = fact[:,:,5:]

    # Prediction
    pred_conf = pred[:,:,0]
    pred_x    = pred[:,:,1]
    pred_y    = pred[:,:,2]
    pred_w    = pred[:,:,3]
    pred_h    = pred[:,:,4]
    pred_cat  = pred[:,:,5:]

    # Mask
    mask_obj = fact_conf
    mask_noobj = 1 - mask_obj

    # --- Confident loss
    conf_loss = K.binary_crossentropy(fact_conf, pred_conf)
    conf_loss = (mask_obj * conf_loss) + (0.015 * mask_noobj * conf_loss)
    # print('conf_loss.shape: ', conf_loss.shape)

    # --- Box loss
    xy_loss  = K.square(fact_x - pred_x) + K.square(fact_y - pred_y)
    wh_loss  = K.square(K.sqrt(fact_w) - K.sqrt(pred_w)) + K.square(K.sqrt(fact_h) - K.sqrt(pred_h))
    box_loss = mask_obj * (xy_loss + wh_loss)
    # print('box_loss.shape: ', box_loss.shape)

    # --- Category loss
    cat_loss = mask_obj * K.sum(K.binary_crossentropy(fact_cat, pred_cat), axis= -1)
    # print('cat_loss.shape: ', cat_loss.shape)

    # --- Total loss
    return K.sum(conf_loss + 5 * box_loss + cat_loss, axis=-1)

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

    return switch(
        tf.equal(tf.math.reduce_mean(precision), 0),
        0.0,
        tf.math.reduce_mean(precision)
        )

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

class HistoryCheckpoint(keras.callbacks.Callback):
    def __init__(self, folder):
        self.folder = folder
    def on_train_begin(self, logs={}):
        with open('{}/model.json'.format(self.folder), 'w') as f:
            json.dump(json.loads(self.model.to_json()), f)
        with open('{}/history.txt'.format(self.folder), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    def on_epoch_end(self, epoch, logs={}):
        keys = ['loss', 'XY_','C_']
        h = ' - '. join(['{}: {:.4f}'.format(k, logs[k]) for k in keys])
        h = h + ' // ' + ' - '. join(['val_{}: {:.4f}'.format(k, logs['val_'+k]) for k in keys])
        h = '{:03d} : '.format(epoch) + h
        with open('{}/history.txt'.format(self.folder), 'a') as f:
            f.write(h + '\n')

def get_model():
    input_layer = Input(shape=(WIDTH, HEIGHT, CHANNEL))
    x = input_layer

    SEED = 2
    for i in range(0, int(math.log(GRID_X/WIDTH, 0.5))):
        SEED = SEED * 2
        x = Conv2D(SEED, 3, padding='same', data_format="channels_last")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2) (x)
        for _ in range(i):
            x = Conv2D(SEED // 2, 1, padding='same', data_format="channels_last")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(SEED , 3, padding='same',data_format="channels_last")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)

    # SEED = SEED * 2
    for i in range(4):
        SEED = SEED // 2
        x = Conv2D(SEED, 1, padding='same', data_format="channels_last")(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.5) (x)


    x = Conv2D(5+len(CLASSES), 1, padding='same', data_format="channels_last")(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(input_layer, x)
    model.compile(optimizer="adam", loss=loss, metrics=[XY_,C_])
    return model

def main():
    model = get_model()

    print(model.summary())
    print('')


    # --- Setup.

    now = datetime.now()
    folder = 'models/{:%Y%m%d-%H%M%S}'.format(now)
    os.makedirs(folder)
    # Callbacks
    history_checkpoint = HistoryCheckpoint(folder=folder)
    model_checkpoint = ModelCheckpoint('{}/model_weights.h5'.format(folder), save_weights_only=True)

    # ---------- Train
    x_val,y_val = load_images_from_directory(validation_path)

    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)

    x_train, y_train = read_pickle_datas('pickles/trains.pickle')

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH,
        epochs=EPOCH,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[model_checkpoint, history_checkpoint])

    # ---------- Test

    x_test,_ = load_images_from_directory(validation_path)

    # x_test = np.concatenate((x_train, x_test),axis=0)

    # Remove the folder
    shutil.rmtree("output_tests/")

    # Create a folder
    directory = "output_tests"
    if not os.path.exists(directory):
        os.makedirs(directory)

    results = model.predict(x_test)

    # Plot training
    for r in range(len(results)):
        x_data = x_test[r]
        y_data = results[r]

        image, labels = convert_data_to_image(x_data, y_data)
        # labels = non_maximum_supression(labels)
        rendered = render_with_labels(image, labels, display = False)
        cv2.imwrite('output_tests/test_render_{:02d}.png'.format(r),rendered)

if __name__ == '__main__':
    main()
