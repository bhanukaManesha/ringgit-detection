#!/usr/bin/env python3
from argparse import ArgumentParser

from lib.data.DataCollection import DataCollection

if __name__ == "__main__" :
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("-r", "-render", dest="render", default= False,
                        help="render each image")
    args = parser.parse_args()

    render = True if args.render else False

    collection = DataCollection.generate()

    collection.save('data/h5py','data.h5')

    if render :
        options = ['train','validation']
        collection.render('output_render', options)
