#!/usr/bin/env python3
import numpy as np
from uuid import uuid4
import cv2, shutil, os
import tensorflow as tf
from datetime import datetime

from common import *
from lib.data.Data import Data

class Render:

    def __init__(self, data, folder, display = False, write = True):
        assert data.dtype == 'batch_data'
        self.folder= folder
        self.display = display
        self.write = write

        self.makedir()
        self.rdata = data


    def apply_nonmaximumsupression(self, dataobj):

        labels = dataobj.y
        remove_index = [0] * len(labels)
        labels = sorted(labels, key=lambda labels: labels[2])

        for i in range(0,len(labels) - 1):

            for j in range(i+1, len(labels)):

                box1 = labels[i]
                box2 = labels[j]

                fp,fx,fy,fw,fh,fc = box1
                pp,px,py,pw,ph,pc = box2

                # IOU
                intersect = (np.minimum(fx+fw, px+pw) - np.maximum(fx, px)) * (np.minimum(fy+fh, py+ph) - np.maximum(fy, py))
                union = (fw * fh) + (pw * ph) - intersect
                value = intersect/union

                if value >= NMS and fp > pp :
                    remove_index[j] = 1
                elif value >= NMS and fp <= pp:
                    remove_index[i] = 1

        new_labels = []
        for i in range(len(remove_index)):
            if remove_index[i] == 0:
                new_labels.append(labels[i])

        dataobj.y = new_labels

    def render_with_labels(self, data):

        for label in data.y:
            # cv2.rectangle(image,(0,0),(5,5),(0,255,0),2)
            cv2.rectangle(data.x, (int(label[1]),int(label[2])), (int(label[1]+label[3]),int(label[2]+label[4])), colors[label[5]], 2)
            # cv2.putText(image, label[5], (label[1], label[2]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[label[5]], 1)

        if self.display:
            cv2.imshow('image',data.x)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.write:
            now = datetime.now()
            filename = 'test_render_{:%H%M%S%f}.png'.format(now)
            cv2.imwrite('{}/{}'.format(self.folder,filename),data.x)

    def makedir(self):
        try:
            # Remove the folder
            shutil.rmtree("{}/".format(self.folder))

        except FileNotFoundError:
            print("Folder not found. Creating new folder.")

        # Create a folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def output_result(self):

        c = len(self.rdata.y) if len(self.rdata.y) <= 100 else 100

        # Plot training
        for r in range(c):

            data = Data((self.rdata.x[r], self.rdata.y[r]), dtype = 'data')

            data.asrender()

            if isNMS:
                self.apply_nonmaximumsupression(data)

            self.render_with_labels(data)
