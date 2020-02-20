#!/usr/bin/env python3
from argparse import ArgumentParser

from lib.data.DataCollection import DataCollection
from lib.data.Render import Render

if __name__ == "__main__" :
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("-r", "-render", dest="render", default= False,
                        help="render each image")
    args = parser.parse_args()

    render = True if args.render else False

    collection = DataCollection.generate()

    collection.write_pickle('data/pickles','collection.pickle')

    if render :
        trainrender = Render(collection.train, 'output_render')
        trainrender.makedir()
        trainrender.output_result()

        valrender = Render(collection.validation, 'output_render')
        valrender.output_result()