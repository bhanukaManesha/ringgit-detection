#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, math
from argparse import ArgumentParser
import numpy as np


from common import *

from lib.yolo.YOLOModel import YOLOModel
from lib.data.DataCollection import DataCollection
from lib.data.Render import Render

def main():

    yolomodel = YOLOModel()

    # Get the data
    datacollection = DataCollection.frompickle('data/pickles', 'collection.pickle')
    yolomodel.datasource = datacollection

    yolomodel.train()

    # ---------- Test

    output_types = ['train','test']

    # Get model prediction
    resultcollection = yolomodel.predict(output_types)

    # Render and write the output

    for otype in output_types:

        trainrender = Render(resultcollection.train, 'output_tests')
        trainrender.makedir()
        trainrender.output_result()

        testrender = Render(resultcollection.test, 'output_tests')
        testrender.output_result()




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
