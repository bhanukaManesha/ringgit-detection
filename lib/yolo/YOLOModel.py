import math

import os
from copy import deepcopy

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from tensorflow.keras.backend import *

from datetime import datetime

from common import *
from lib.yolo.YOLOMetrics import YOLOMetrics

class YOLOModel :

    def __init__(self, options):
        self._model = self.get_model()
        print(self._model.summary())
        self._metrics = YOLOMetrics()

        # Setup the checkpoints
        now = datetime.now()
        folder = 'models/{:%Y%m%d-%H%M%S}'.format(now)
        os.makedirs(folder)

        self._history_checkpoint = YOLOMetrics.HistoryCheckpoint(folder=folder)
        self._model_checkpoint = ModelCheckpoint('{}/model_weights.h5'.format(folder), save_weights_only=True)

        self._datasource = None
        self._options = options

    def get_model(self):
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
        metrics = YOLOMetrics()
        model.compile(optimizer="adam", loss=self.loss, metrics=[metrics.XY_,metrics.C_])
        return model

    def load_model(self):

        directory = "models/"
        folders = [x[0] for x in os.walk(directory)]
        folders.sort()
        model_path = folders[-1]

        try:
            with open("{}/model.json".format(model_path)) as json_file:
                json_config = json_file.read()

            self.model = keras.models.model_from_json(json_config)
            self.model.load_weights("{}/model_weights.h5".format(model_path))

        except FileNotFoundError:
            print("Check the file path.")

    def loss(self,fact, pred):
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

    def train(self):
        self._model.fit(
            x=self._datasource.train.x,
            y=self._datasource.train.y,
            batch_size=self._options['batch'],
            epochs=self._options['epoch'],
            validation_data=(self._datasource.validation.x, self._datasource.validation.y),
            shuffle=True,
            callbacks=[self._model_checkpoint, self._history_checkpoint])

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