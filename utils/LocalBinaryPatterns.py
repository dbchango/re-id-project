import cv2
from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, radius):
        # store the number of points and radius
        self.numPoints = 8 * radius
        self.radius = radius

    def lbp(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, self.numPoints, self.radius, method='uniform')

        return lbp

    def describe(self, image, eps=1e-7):
        lbp_image = self.lbp(image)

        (hist, _) = np.histogram(lbp_image.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        # hist, bins = np.histogram(lbp_image.ravel(), 256, [0, 256])
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist