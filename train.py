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
    datacollection = DataCollection.fromh5py('data/h5py', 'data.h5')
    yolomodel._datasource = datacollection

    # HP_SEED = hp.HParam('seed', hp.Discrete([16]))
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['nadam']))
    # HP_SEED = hp.HParam('seed', hp.Discrete([8,16,32]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['nadam']))

    try:
        # Remove the folder
        shutil.rmtree("{}/".format('logs'))
    except FileNotFoundError:
        pass

    # Create a folder
    if not os.path.exists('logs'):
        os.makedirs('logs')

    session_num = 0

    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            'optimizer': optimizer,
        }
        run_name = "run-%s" % {h: hparams[h] for h in hparams}
        print('--- Starting trial: %s' % run_name)
        print({h: hparams[h] for h in hparams})
        yolomodel.train('logs/hparam_tuning/' + run_name, hparams)


        # ---------- Test

        renderoptions = ['train','validation','test']

        # Get model prediction
        resultcollection = yolomodel.predict(renderoptions)

        # Render the result
        resultcollection.render('output_tests/{}'.format(run_name), renderoptions)

        session_num += 1


    



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
