#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, math
from argparse import ArgumentParser
import numpy as np

from data import Data
from common import *
from model import YOLOModel
from render import Render


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():

    yolomodel = YOLOModel()
    print(yolomodel.model.summary())

    data = Data()

    # ---------- Train
    x_train, y_train = data.read_pickle_datas()

    x_val,y_val = data.load_images_from_directory()
    x_val,y_val = np.asarray(x_val) , np.asarray(y_val)


    yolomodel.model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH,
        epochs=EPOCH,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[yolomodel.model_checkpoint, yolomodel.history_checkpoint])

    # ---------- Test
    x_test,_ = data.load_images_from_directory()

    # Remove the folder
    shutil.rmtree("output_tests/")

    # Create a folder
    directory = "output_tests"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get model prediction
    results = yolomodel.model.predict(x_test)

    # Render and write the output
    r = Render()
    r.output_result(x_test,results)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-e", "-epochs", dest="epochs", default= 100,
                        help="number of epochs", metavar="file")
    parser.add_argument("-b", "-batch_size", dest="batch_size", default= 8,
                        help="render each image")
    args = parser.parse_args()

    EPOCH = int(args.epochs)
    BATCH = int(args.batch_size)

    main()
