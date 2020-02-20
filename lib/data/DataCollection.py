import glob, pathlib, cv2, json, pickle
import numpy as np
import os

from common import *

from lib.data.Data import Data
from lib.data.DataGenerator import DataGenerator

class DataCollection:

    def __init__(self, train, validation, test):

        self.train = train
        self.validation = validation
        self.test = test

    @classmethod
    def frompickle(cls, folder, picklename):
        with open('{}/{}'.format(folder, picklename), 'rb') as f:
            col = pickle.load(f)

        return cls(col.train, col.validation, col.test)


    @classmethod
    def generate(cls):
        '''
        main function to build the data colllection
        '''
        generator = DataGenerator()

        # traininig data
        train = next(generator.serve(SAMPLE))

        # validatation data
        validation = generator.load_images_from_directory('data/val')

        # testing data
        test = generator.load_images_from_directory('data/val')

        return cls(train, validation, test)

    def write_pickle(self, folder, picklename):

        # Create a folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open('{}/{}'.format(folder, picklename), 'wb') as f:
            pickle.dump(self, f)

    def render_collection(self):
