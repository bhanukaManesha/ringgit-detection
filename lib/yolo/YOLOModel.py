import math

import os
from copy import deepcopy

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, Dropout, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.backend import *
from tensorflow.keras.losses import *
from tensorflow.keras import regularizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from datetime import datetime

from common import *
from lib.yolo.YOLOMetrics import YOLOMetrics

from tensorboard.plugins.hparams import api as hp

class YOLOModel :

    def __init__(self, options=None, loadmodel = False):
        self._metrics = YOLOMetrics()
        self._model = None

        if loadmodel:
            self.load_model(MODELPATH)
        
        else:
            # Setup the checkpoints
            now = datetime.now()
            self.folder = 'models/{:%Y%m%d-%H%M%S}'.format(now)
            os.makedirs(self.folder)

        # self._history_checkpoint = YOLOMetrics.HistoryCheckpointCallback(folder=folder)
        # self._tensorboard = YOLOMetrics.TensorboardCallback()
        # self._earlystopping = YOLOMetrics.EarlyStoppingCallback()
        # self._model_checkpoint = ModelCheckpoint('{}/model_weights.h5'.format(folder), save_weights_only=True)


        self._datasource = None
        self._options = options

    def get_model(self,hparams):

        input_layer = Input(shape=(WIDTH, HEIGHT, CHANNEL))
        x = input_layer

        SEED = hparams['seed']
        for i in range(0, int(math.log(GRID_X/WIDTH, 0.5))):
            SEED = SEED * 2
            x = Conv2D(SEED, 3, padding='same', data_format="channels_last",kernel_regularizer=regularizers.l2(0.01))(x)
            # x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # x = Dropout(0.2) (x)
            for _ in range(i):
                x = Conv2D(SEED // 2, 1, padding='same', data_format="channels_last", kernel_regularizer=regularizers.l2(0.01))(x)
                # x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(SEED , 3, padding='same',data_format="channels_last",kernel_regularizer=regularizers.l2(0.01))(x)
                # x = BatchNormalization()(x)
                x = Activation('relu')(x)

            # x = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(x)
            x = Conv2D(SEED , 2, strides=2, padding='same',data_format="channels_last",kernel_regularizer=regularizers.l2(0.01))(x)

        SEED = SEED * 2
        for i in range(4):
            SEED = SEED // 2
            x = Conv2D(SEED, 1, padding='same', data_format="channels_last", kernel_regularizer=regularizers.l2(0.01))(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
            # x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # x = Dropout(0.2) (x)

        x = Conv2D(5+len(CLASSES), 1, padding='same', data_format="channels_last",kernel_regularizer=regularizers.l2(0.01))(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
        # x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)

        model = Model(input_layer, x)
        metrics = YOLOMetrics()
        model.compile(optimizer=hparams['optimizer'], loss=self.loss, metrics=[YOLOMetrics.P_, YOLOMetrics.XY_,YOLOMetrics.C_])
        return model

    def get_mobilenetv2(self,hparams):
        base_model = MobileNetV2(input_shape=(WIDTH, HEIGHT, CHANNEL), weights="imagenet", include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
        # for layer in base_model.layers:
        #     layer.trainable=False
        x = base_model.output

        SEED = 128
        for i in range(4):
            SEED = SEED // 2
            x = Conv2D(SEED, 1, padding='same', data_format="channels_last", kernel_regularizer=regularizers.l2(0.01))(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2) (x)
            x = Dropout(0.1) (x)

        x = Conv2D(5+len(CLASSES), 1, padding='same', data_format="channels_last",kernel_regularizer=regularizers.l2(0.01))(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
        x = Activation('sigmoid')(x)


        model = Model(base_model.input, x)
        metrics = YOLOMetrics()
        model.compile(optimizer=hparams['optimizer'], loss=self.loss, metrics=[YOLOMetrics.P_, YOLOMetrics.XY_,YOLOMetrics.C_])
        return model


    def load_model(self, model_path):

        if model_path == None:
            directory = "models/"
            folders = [x[0] for x in os.walk(directory)]
            folders.sort()
            model_path = folders[-1]
        
        print(model_path)

        try:
            with open("{}/model.json".format(model_path)) as json_file:
                json_config = json_file.read()

            self._model = model_from_json(json_config)
            self._model.load_weights("{}/model_weights.h5".format(model_path))

        except FileNotFoundError:
            print("Check the file path.")


    def loss(self, fact, pred):
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
        mask_noobj = 1 - fact_conf

        # --- Confident loss
        conf_loss = K.binary_crossentropy(fact_conf, pred_conf)
        conf_loss = (mask_obj * conf_loss) + (0.02 * mask_noobj * conf_loss)
        # print('conf_loss.shape: ', conf_loss.shape)

        # --- Box loss
        # xy_loss  = K.square(fact_x - pred_x) + K.square(fact_y - pred_y)
        xy_loss = K.binary_crossentropy(fact_x,pred_x) + K.binary_crossentropy(fact_y,pred_y)
        # wh_loss  = K.square(K.sqrt(fact_w) - K.sqrt(pred_w)) + K.square(K.sqrt(fact_h) - K.sqrt(pred_h))
        wh_loss = K.binary_crossentropy(fact_w,pred_w) + K.binary_crossentropy(fact_y,pred_y)
        box_loss = mask_obj * (xy_loss + wh_loss)
        # print('box_loss.shape: ', box_loss.shape)

        # --- Category loss
        cat_loss = mask_obj * K.sum(K.binary_crossentropy(fact_cat, pred_cat),axis=-1)
        # print('cat_loss.shape: ', cat_loss.shape)

        # --- Total loss
        total_loss =  K.sum(conf_loss + 20 * box_loss + cat_loss, axis=-1)
        # print('total_loss.shape: ', total_loss.shape)

        return total_loss

    # def loss(self,fact, pred):
    #     fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])
    #     pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(CLASSES)])

    #     # Truth
    #     fact_conf = fact[:,:,0]
    #     fact_x    = fact[:,:,1]
    #     fact_y    = fact[:,:,2]
    #     fact_w    = fact[:,:,3]
    #     fact_h    = fact[:,:,4]
    #     fact_cat  = fact[:,:,5:]

    #     # Prediction
    #     pred_conf = pred[:,:,0]
    #     pred_x    = pred[:,:,1]
    #     pred_y    = pred[:,:,2]
    #     pred_w    = pred[:,:,3]
    #     pred_h    = pred[:,:,4]
    #     pred_cat  = pred[:,:,5:]

    #     # Mask
    #     mask_obj = fact_conf
    #     mask_noobj = 1 - fact_conf

    #     # --- Confident loss
    #     conf_loss = K.square(fact_conf - pred_conf)
    #     conf_loss = K.sum((mask_obj + 0.01 * mask_noobj) * conf_loss, axis = -1)
    #     # print('conf_loss.shape: ', conf_loss.shape)

    #     # --- Box loss
    #     xy_loss  = K.square(fact_x - pred_x) + K.square(fact_y - pred_y)
    #     wh_loss  = K.square(K.sqrt(fact_w) - K.sqrt(pred_w)) + K.square(K.sqrt(fact_h) - K.sqrt(pred_h))
    #     box_loss = 50 * K.sum(mask_obj * (xy_loss + wh_loss), axis= -1)
    #     # print('box_loss.shape: ', box_loss.shape)

    #     # --- Category loss
    #     cat_loss = K.sum(mask_obj * K.sum(K.square(fact_cat - pred_cat),axis=-1), axis= -1)
    #     # print('cat_loss.shape: ', cat_loss.shape)

    #     # --- Total loss
    #     return conf_loss + box_loss + cat_loss

        
    def train(self, logdir, hparams):
        self._model = self.get_mobilenetv2(hparams)
        print(self._model.summary())

        self._model.fit(
            x=self._datasource.train.x,
            y=self._datasource.train.y,
            batch_size=self._options['batch'],
            epochs=self._options['epoch'],
            validation_data=(self._datasource.validation.x, self._datasource.validation.y),
            shuffle=True,
            callbacks=[
                ModelCheckpoint('{}/model_weights.h5'.format(self.folder), save_weights_only=True),
                YOLOMetrics.HistoryCheckpointCallback(folder=self.folder),
                YOLOMetrics.TensorboardCallback(logdir),
                hp.KerasCallback(logdir, hparams)])

    def predict(self, options):

        collection = deepcopy(self._datasource)

        for option in options:

            if option == 'train':
                collection.train.y = self._model.predict(collection.train.x)

            elif option == 'validation':
                collection.validation.y = self._model.predict(collection.validation.x)

            elif option == 'test':
                collection.test.y = self._model.predict(collection.test.x)

            else:
                print('Invalid option.')

        return collection