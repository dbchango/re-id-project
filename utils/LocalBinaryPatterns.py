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

    def describe(self, image):
        lbp_image = self.lbp(image)
        n_bins = int(lbp_image.max() + 1)
        (hist, _) = np.histogram(lbp_image.ravel(), n_bins, [0, n_bins], density=True)
        # hist, bins = np.histogram(lbp_image.ravel(), 256, [0, 256])
        return hist