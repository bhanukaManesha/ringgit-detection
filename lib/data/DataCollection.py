import glob, pathlib, cv2, json, pickle
import numpy as np
import os

from common import *

from lib.data.Data import Data
from lib.data.DataGenerator import DataGenerator
from lib.data.Render import Render

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
        # validation = generator.from_directory('data/val', "jpeg")
        # validation = next(generator.serve(10))
        validation = train

        # testing data
        test = generator.from_directory('data/val', "jpeg")

        return cls(train, validation, test)

    def write_pickle(self, folder, picklename):

        # Create a folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open('{}/{}'.format(folder, picklename), 'wb') as f:
            pickle.dump(self, f)

    def render(self, folder, options):
        # Render and write the output
        for option in options:

            if option == 'train':
                assert self.train != None
                trainrender = Render(self.train, '{}/{}'.format(folder, option))
                trainrender.output_result()

            elif option == 'validation':
                assert self.validation != None
                trainrender = Render(self.validation, '{}/{}'.format(folder, option))
                trainrender.output_result()

            elif option == 'test':
                assert self.test != None
                trainrender = Render(self.test, '{}/{}'.format(folder, option))
                trainrender.output_result()

            else:
                print('Invalid option.')
