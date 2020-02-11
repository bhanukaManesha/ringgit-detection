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
tf.executing_eagerly()

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow import keras
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from tensorflow.keras.backend import *

from rotation_generator import generator

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    # print('conf', conf_loss)
    conf_loss = (mask_obj * conf_loss) + (mask_noobj * conf_loss)
    
    # --- Box loss
    xy_loss  = tf.square(fact_x - pred_x) + tf.square(fact_y - pred_y)
    wh_loss  = tf.square(tf.sqrt(fact_w) - tf.sqrt(pred_w)) + tf.square(tf.sqrt(fact_h) - tf.sqrt(pred_h))
    box_loss = 5 * mask_obj * (xy_loss + wh_loss) 
    # print('box_loss.shape: ', box_loss.shape)

    # --- Category loss
    cat_loss = mask_obj * sum(tf.square(fact_cat - pred_cat), axis=-1)
    # print('cat_loss.shape: ', cat_loss.shape)    

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
    return binary_accuracy(fact_conf, pred_conf,threshold=0.8)

def Precision_(fact,pred):
    iou = calculate_iou(fact,pred)

    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])


    fact_conf = fact[:,:,0]
    pred_conf = pred[:,:,0]

    # total = tf.math.count_nonzero(fact_conf, dtype=tf.float32)

    detection = tf.cast(pred_conf >= 0.75, dtype=tf.float32)
    no_detection = tf.cast(pred_conf < 0.75, dtype=tf.float32)

    iou_greater = tf.cast(iou >= 0.75, dtype=tf.float32)
    iou_less = tf.cast(iou < 0.75, dtype=tf.float32)

    tp = tf.math.count_nonzero(detection * iou_greater, dtype=tf.float32, axis=1)
    fp = tf.math.count_nonzero(detection * iou_less, dtype=tf.float32, axis=1)

    fn = tf.math.count_nonzero(no_detection * fact_conf, dtype=tf.float32, axis=1) 

    precision = (tp / (tp + fp))
    
    return switch(tf.equal(tf.math.reduce_mean(precision), 0),0.0,tf.math.reduce_mean(precision))

def Recall_(fact,pred):
    iou = calculate_iou(fact,pred)

    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])


    fact_conf = fact[:,:,0]
    pred_conf = pred[:,:,0]

    detection = tf.cast(pred_conf >= 0.75, dtype=tf.float32)
    no_detection = tf.cast(pred_conf < 0.75, dtype=tf.float32)

    iou_greater = tf.cast(iou >= 0.75, dtype=tf.float32)
    iou_less = tf.cast(iou < 0.75, dtype=tf.float32)

    tp = tf.math.count_nonzero(detection * iou_greater, dtype=tf.float32, axis=1)

    fp = tf.math.count_nonzero(detection * iou_less, dtype=tf.float32, axis=1)

    fn = tf.math.count_nonzero(no_detection * fact_conf, dtype=tf.float32, axis=1) 
    
    total = tf.math.count_nonzero(fact_conf, dtype=tf.float32, axis=1)

    recall = tp / total

    # print('cat_loss.shape: ', precision.shape)    
    
    return switch(tf.equal(tf.math.reduce_mean(recall), 0),0.0,tf.math.reduce_mean(recall))

def XY_(fact, pred):
    fact = tf.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    pred = tf.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    # Truth
    fact_conf = fact[:,:,0]
    fw = fact[:,:,3] * WIDTH
    fh = fact[:,:,4] * HEIGHT
    fx = fact[:,:,0] * GRID_WIDTH - fw/2
    fy = fact[:,:,1] * GRID_HEIGHT - fh/2
    # Prediction
    pw = pred[:,:,3] * WIDTH
    ph = pred[:,:,4] * HEIGHT
    px = pred[:,:,0] * GRID_WIDTH - pw/2
    py = pred[:,:,1] * GRID_HEIGHT - ph/2
    # IOU
    intersect = (tf.minimum(fx+fw, px+pw) - tf.maximum(fx, px)) * (tf.minimum(fy+fh, py+ph) - tf.maximum(fy, py))
    union = (fw * fh) + (pw * ph) - intersect
    nonzero_count = tf.math.count_nonzero(fact_conf, dtype=tf.float32, axis=1)

    # print('nonzero: ', nonzero_count.shape) 
    o = sum((intersect / union) * fact_conf , axis = 1) / nonzero_count
    h = tf.math.reduce_mean(o)

    return switch(tf.equal(h, 0),0.0,h)


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
        keys = ['loss', 'Precision_', 'Recall_','XY_']
        h = ' - '. join(['{}: {:.4f}'.format(k, logs[k]) for k in keys])
        h = h + ' // ' + ' - '. join(['val_{}: {:.4f}'.format(k, logs['val_'+k]) for k in keys])
        h = '{:03d} : '.format(epoch) + h
        with open('{}/history.txt'.format(self.folder), 'a') as f:
            f.write(h + '\n')



def get_model():
    input_layer = Input(shape=(WIDTH, HEIGHT, CHANNEL))
    x = input_layer

    SEED = 4
    for i in range(0, int(math.log(GRID_X/WIDTH, 0.5))):
        SEED = SEED * 3
        x = Conv2D(SEED, 3, padding='same', data_format="channels_last", kernel_initializer='he_uniform', bias_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # for _ in range(i):
        #     x = Conv2D(SEED // 2, 1, padding='same', data_format="channels_last")(x)
        #     x = BatchNormalization()(x)
        #     x = Activation('relu')(x)

        #     x = Conv2D(SEED , 3, padding='same',data_format="channels_last")(x)
        #     x = BatchNormalization()(x)
        #     x = Activation('relu')(x)
            
        x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)

    
    SEED = SEED * 2
    for i in range(2):
        SEED = SEED // 2
        x = Conv2D(SEED, 1, padding='same', data_format="channels_last", kernel_initializer='he_uniform', bias_initializer='he_uniform')(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(5+len(CLASSES), 1, padding='same', data_format="channels_last", kernel_initializer='he_uniform', bias_initializer='he_uniform')(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(input_layer, x)

    # sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)

    model.compile(optimizer="adam", loss=loss, metrics=[Precision_,Recall_,XY_])
    return model

def load_images_from_directory(path):

    image_paths = glob.glob(path + "images/*.jpg")
    label_paths = glob.glob(path + "labels/*.txt")

    image_paths.sort()
    label_paths.sort()

    x_train = []
    y_train = []

    
    for path in image_paths:
        # print(path)
        image = cv2.imread(path).astype(np.float32)
        x_train.append(image)


    for path in label_paths:
        # print(path)
        with open(path) as f:
            annotations = f.readlines()

        annotations = [x.strip() for x in annotations]
        annotations = [x.split() for x in annotations]
        annotations = np.asarray(annotations)

        y_data = np.zeros((GRID_Y, GRID_X, 5+len(CLASSES)))

        for row in range(GRID_X):
            for col in range(GRID_Y):
                y_data[row, col, 0] = float(annotations[row * GRID_X + col][0])
                y_data[row, col, 1:5] = [
                    float(annotations[row * GRID_X + col][2]),
                    float(annotations[row * GRID_X + col][3]),
                    float(annotations[row * GRID_X + col][4]),
                    float(annotations[row * GRID_X + col][5])
                ]
                y_data[row, col, int(5+float(annotations[row * GRID_X + col][1]))] = float(1)

        y_train.append(y_data)

    return x_train,y_train

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
    x_test, y_test = next(generator(32))

    real_x_train,real_y_train = load_images_from_directory("real/")
    x_val = np.concatenate((np.asarray(real_x_train),np.asarray(x_test)), axis=0)
    y_val = np.concatenate((np.asarray(real_y_train),np.asarray(y_test)), axis=0)


    model.fit(
        x=generator(BATCH),
        steps_per_epoch=(SAMPLE // BATCH),
        epochs=EPOCH,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[model_checkpoint, history_checkpoint])
    
    # ---------- Test

    # Remove the folder
    shutil.rmtree("output_tests/")
    
    # Create a folder
    directory = "output_tests"
    if not os.path.exists(directory):
        os.makedirs(directory)
    

    results = model.predict(x_val)

    # Plot training
    for r in range(len(results)):
        x_data = x_val[r]
        y_data = results[r]
        # y_data = y_val[r]

        image, texts = convert_data_to_image(x_data, y_data)
        print(len(texts))
        texts = non_maximum_supression(texts)
        rendered = render_with_labels(image, texts, display = False)
        cv2.imwrite('output_tests/test_render_{:02d}.png'.format(r),rendered)

if __name__ == '__main__':
    main()

    # k = [
    #     [0.9520847, 11, 6, 41, 38, 'RM50'],
    #     [0.9994442, 21, 2, 37, 44, 'RM50'],
    #     [0.87747, 28, 3, 41, 45, 'RM50'],
    #     [0.99999964, 8, 10, 48, 43, 'RM50'],
    #     [0.99628925, 0, 19, 47, 42, 'RM50'],
    #     [0.81187415, 4, 28, 43, 42, 'RM50'],
    #     [0.8049084, 3, 28, 44, 42, 'RM50'],
    #     [0.9836606, 9, 28, 45, 39, 'RM50'],
    #     [0.9574118, 18, 30, 44, 37, 'RM50'],
    #     [0.8192027, 11, 44, 41, 26, 'RM50']
    # ]
    
    # non_maximum_supression(k)


