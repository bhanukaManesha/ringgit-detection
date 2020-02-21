#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, math
from argparse import ArgumentParser
import numpy as np


from common import *

from lib.yolo.YOLOModel import YOLOModel
from lib.data.DataCollection import DataCollection

def main(options):
    yolomodel = YOLOModel(options)

    # Get the data
    datacollection = DataCollection.frompickle('data/pickles', 'collection.pickle')
    yolomodel._datasource = datacollection

    yolomodel.train()

    # ---------- Test

    options = ['train','test']

    # Get model prediction
    resultcollection = yolomodel.predict(options)

    # Render the result
    resultcollection.render('output_tests', options)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-e", "-epochs", dest="epochs", default= 100,
                        help="number of epochs", metavar="file")
    parser.add_argument("-b", "-batch_size", dest="batch_size", default= 8,
                        help="render each image")
    args = parser.parse_args()

    options = {
        'epoch' : int(args.epochs),
        'batch' : int(args.batch_size)
    }

    main(options)
