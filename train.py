from __future__ import absolute_import, division, print_function, unicode_literals

from common import *
from generate_data import *

from datetime import datetime
import os
import math
import numpy as np
import cv2
import json

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow import keras
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.backend import *


def loss(fact, pred):
    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])

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
    conf_loss = tf.square(fact_conf - pred_conf)
    conf_loss = (mask_obj * conf_loss) + (mask_noobj * conf_loss)
    print('conf_loss.shape: ', conf_loss.shape)

    # --- Box loss
    xy_loss  = tf.square(fact_x - pred_x) + tf.square(fact_y - pred_y)
    wh_loss  = tf.square(tf.sqrt(fact_w) - tf.sqrt(pred_w)) + tf.square(tf.sqrt(fact_h) - tf.sqrt(pred_h))
    box_loss = mask_obj * (xy_loss + wh_loss)
    print('box_loss.shape: ', box_loss.shape)

    # --- Category loss
    cat_loss = mask_obj * sum(tf.square(fact_cat - pred_cat), axis=-1)
    print('cat_loss.shape: ', cat_loss.shape)

    # --- Total loss
    return sum(conf_loss + box_loss + cat_loss, axis=-1)

def P_(fact, pred):
    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    # Truth
    fact_conf = fact[:,:,0]
    # Prediction
    pred_conf = pred[:,:,0]
    # PROBABILITY
    return binary_accuracy(fact_conf, pred_conf)

def XY_(fact, pred):
    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
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
    intersect = (tf.minimum(fx+fw, px+pw) - tf.maximum(fx, px)) * (tf.minimum(fy+fh, py+ph) - tf.maximum(fy, py))
    union = (fw * fh) + (pw * ph) - intersect
    nonzero_count = tf.math.count_nonzero(fact_conf, dtype=tf.float32)
    return switch(
        tf.equal(nonzero_count, 0),
        1.0,
        sum((intersect / union) * fact_conf) / nonzero_count
    )

def C_(fact, pred):
    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    # Truth
    fact_conf = fact[:,:,0]
    fact_cat = fact[:,:,5:]
    # Prediction
    pred_cat = pred[:,:,5:]
    # CLASSIFICATION
    nonzero_count = tf.math.count_nonzero(fact_conf, dtype=tf.float32)
    return switch(
        tf.equal(nonzero_count, 0),
        1.0,
        sum(categorical_accuracy(fact_cat, pred_cat) * fact_conf) / nonzero_count
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
        keys = ['loss', 'P_', 'XY_', 'C_']
        h = ' - '. join(['{}: {:.4f}'.format(k, logs[k]) for k in keys])
        h = h + ' // ' + ' - '. join(['val_{}: {:.4f}'.format(k, logs['val_'+k]) for k in keys])
        h = '{:03d} : '.format(epoch) + h
        with open('{}/history.txt'.format(self.folder), 'a') as f:
            f.write(h + '\n')



def get_model():
    input_layer = Input(shape=(WIDTH, HEIGHT, CHANNEL))
    x = input_layer

    SEED = 32

    for i in range(0, int(math.log(GRID_X/WIDTH, 0.5))):
        SEED = SEED * 2
        x = Conv2D(SEED, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        for _ in range(i):
            x = Conv2D(SEED // 2, 1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(SEED , 3, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D()(x)

    SEED = SEED * 2
    for i in range(5):
        SEED = SEED // 2
        x = Conv2D(SEED, 1, padding='same')(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(5+len(CLASSES), 1, padding='same')(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(input_layer, x)
    # model.compile(optimizer=R]Adam(), loss=loss, metrics=[P_, XY_, C_])
    model.compile(optimizer='adam', loss=loss, metrics=[P_, XY_, C_])
    return model


def generator(batch_size, test=True):

    if not test:
        images, labels = read_data(test=False)
    else:
        images, labels = read_data(test=True)

    while True:
        # Empty batch arrays.
        x_trains = []
        y_trains = []
        # Create batch data.
        for i in range(batch_size):
            # image, texts = generate_image(WIDTH, HEIGHT, seeds=random.sample(TEXTS, k=len(TEXTS)))
            x_data, y_data = load_image(images, labels)

            # if not texts:
            #     image = generate_noise(image)
            # x_data, y_data = convert_image_to_data(image, texts)


            # Append
            x_trains.append(x_data)
            y_trains.append(y_data)

        x_trains = np.asarray(x_trains).reshape((batch_size, HEIGHT, WIDTH, CHANNEL))
        y_trains = np.asarray(y_trains).reshape((batch_size, GRID_Y, GRID_X, 5+len(CLASSES)))
        yield x_trains, y_trains



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

    SAMPLE = 200
    BATCH  = 8
    EPOCH  = 10

    x_vals, y_vals = next(generator(32, test=False))

    model.fit_generator(
        generator=generator(BATCH, test=False),
        steps_per_epoch=(SAMPLE // BATCH),
        epochs=EPOCH,
        validation_data=(x_vals, y_vals),
        shuffle=True,
        callbacks=[model_checkpoint, history_checkpoint])

    # # ---------- Test

    x_tests, y_tests = next(generator(10, test=True))

    # results = y_tests
    results = model.predict(x_tests)

    for r in range(len(results)):
        x_data = x_tests[r]
        y_data = results[r]

        image, texts = convert_data_to_image(x_data, y_data)
        rendered = render_with_labels(image, texts)
        cv2.imwrite('output_tests/test_render_{:02d}.png'.format(r),ren)
        # rendered.save('output_tests/test_render_{:02d}.png'.format(r), 'PNG')


if __name__ == '__main__':
    main()
