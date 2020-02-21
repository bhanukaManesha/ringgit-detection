#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

from lib.yolo.YOLOModel import YOLOModel

def main():

    yolomodel = YOLOModel()
    yolomodel.load_model()
    print(yolomodel.model.summary())

    # Fix this asap
    yolodata = Data()
    x_test,y_test = yolodata.get_test_data()

    results = yolomodel.model.predict(x_test)

    r = Render()
    r.output_result(x_test, results)

if __name__ == "__main__":

    print("Initializing...")
    main()
    print("Test Done.")
