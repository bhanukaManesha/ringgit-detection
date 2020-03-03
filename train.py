#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os, shutil, math
from argparse import ArgumentParser
import numpy as np


from common import *

from lib.yolo.YOLOModel import YOLOModel
from lib.data.DataCollection import DataCollection

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

def main(options):
    yolomodel = YOLOModel(options)

    # Get the data
    datacollection = DataCollection.frompickle('data/pickles', 'collection.pickle')
    yolomodel._datasource = datacollection

    HP_SEED = hp.HParam('seed', hp.Discrete([2,4,8,16]))
    HP_EXTEND = hp.HParam('extend', hp.Discrete([0,1]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','nadam']))

    # Remove the folder
    shutil.rmtree("{}/".format('logs'))
    # Create a folder
    if not os.path.exists('logs'):
        os.makedirs('logs')

    session_num = 0

    for seed in HP_SEED.domain.values:
        for extend in HP_EXTEND.domain.values:
            for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                for optimizer in HP_OPTIMIZER.domain.values:
                    hparams = {
                        'seed': seed,
                        'extend' : extend,
                        'dropout': dropout_rate,
                        'optimizer': optimizer,
                    }
                    run_name = "run-%s" % {h: hparams[h] for h in hparams}
                    print('--- Starting trial: %s' % run_name)
                    print({h: hparams[h] for h in hparams})
                    yolomodel.train('logs/hparam_tuning/' + run_name, hparams)
                    session_num += 1


    # ---------- Test

    # options = ['train','test']

    # Get model prediction
    # resultcollection = yolomodel.predict(options)

    # Render the result
    # resultcollection.render('output_tests', options)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-e", "-epochs", dest="epochs", default= 2,
                        help="number of epochs", metavar="file")
    parser.add_argument("-b", "-batch_size", dest="batch_size", default= 8,
                        help="render each image")
    args = parser.parse_args()

    options = {
        'epoch' : int(args.epochs),
        'batch' : int(args.batch_size)
    }

    if int(args.epochs) == 0:
        options['epoch'] = EPOCH

    main(options)
