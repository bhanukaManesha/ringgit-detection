#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

from model import YOLOModel
from render import Render
from data import Data

def main():

    yolomodel = YOLOModel()
    yolomodel.load_model()
    print(yolomodel.model.summary())

    # Fix this asap
    data = Data()
    x_test,y_test = data.load_images_from_directory()
    x_test, y_test = np.asarray(x_test), np.asarray(y_test)

    results = yolomodel.model.predict(x_test)

    r = Render()
    r.output_result(x_test, results)



if __name__ == "__main__":

    print("Initializing...")
    main()
    print("Test Done.")
