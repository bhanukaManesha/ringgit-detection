#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

from lib.yolo.YOLOModel import YOLOModel
from lib.data.DataCollection import DataCollection
from lib.data.Render import Render

def main():

    yolomodel = YOLOModel()
    yolomodel.load_model()
    print(yolomodel.model.summary())

    # Get the data
    datacollection = DataCollection.fromh5py('data/h5py', 'data.h5')
    yolomodel._datasource = datacollection

    # ---------- Test

    renderoptions = ['test']

    # Get model prediction
    resultcollection = yolomodel.predict(renderoptions)

    # Render the result
    resultcollection.render('output_tests/{}'.format("test"), renderoptions)


if __name__ == "__main__":

    print("Initializing...")
    main()
    print("Test Done.")
