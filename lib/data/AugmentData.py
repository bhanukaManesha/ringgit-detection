import cv2
import numpy as np
import random

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from lib.data.Data import Data
from common import *

class AugmentData(Data):

    def __init__(self, data, dtype):
        super().__init__(data, dtype)
        self.augmentationseq = iaa.Sequential([
            iaa.ChangeColorTemperature((4000, 9000)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Rotate(rotate=(0, 359)),
            # iaa.Affine(scale=(0.8, 1.2)),
            # iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.5),
            # iaa.PerspectiveTransform(scale=(0.02, 0.08), keep_size=True),
            iaa.Resize({'width': WIDTH, 'height': HEIGHT}, interpolation=imgaug.ALL)
        ])

    @classmethod
    def fromdataobj(cls, dataobj):
        return cls(data=(dataobj.x, dataobj.y), dtype=dataobj.dtype)

    def augmentation(self):

        polygons = []
        for polygon in self.y:
            polygons.append(Polygon(polygon['points']))
            polygon['points'] = []

        pps = PolygonsOnImage(polygons, shape=self.x.shape)

        # Augment.
        self.x, pps = self.augmentationseq(image=self.x, polygons=pps)

        for i,polygon in enumerate(self.y):

            for (x,y) in pps[i]:

                if x <= 0:
                    x = 0
                if y <= 0:
                    y = 0
                if x >= WIDTH:
                    x = WIDTH - 1
                if y >= HEIGHT:
                    y = HEIGHT - 1

                polygon['points'].append([x,y])
