import glob, pathlib, cv2, json, pickle
import numpy as np
import os
import h5py

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
    def fromh5py(cls, folder, h5name):
        h5f = h5py.File('{}/{}'.format(folder, h5name),'r')
        # train
        a = Data(
            (h5f['train_x'][:],h5f['train_y'][:]),
            "batch_data"
            )

        b = Data(
            (h5f['validation_x'][:],h5f['validation_y'][:]),
            "batch_data"
            )

        c = Data(
            (h5f['test_x'][:],h5f['test_y'][:]),
            "batch_data"
            )

        h5f.close()

        return cls(a, b, c)

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
        validation = generator.from_directory('data/val', "jpeg")
        # validation = next(generator.serve(32))
        # validation = train

        # testing data
        test = generator.from_directory('data/val', "jpeg")

        return cls(train, validation, test)

    def save(self,folder,h5name):
        
        # Create a folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        h5f = h5py.File('{}/{}'.format(folder, h5name), 'w')

        # train
        h5f.create_dataset('train_x', data=self.train.x)
        h5f.create_dataset('train_y', data=self.train.y)

        # validation
        h5f.create_dataset('validation_x', data=self.validation.x)
        h5f.create_dataset('validation_y', data=self.validation.y)

        # test
        h5f.create_dataset('test_x', data=self.test.x)
        h5f.create_dataset('test_y', data=self.test.y)

        h5f.close()

    def write_pickle(self, folder, picklename):

        # Create a folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open('{}/{}'.format(folder, picklename), 'wb') as f:
            pickle.dump(self, f, protocol=4)

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
