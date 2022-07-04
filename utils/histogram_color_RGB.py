import cv2


class RGBHistogram(object):
    def __init__(self, bins):
        """
        Instantiates an object of this class.
        :param bins: Number of bins (per channel) used to compute the color histogram of images.
        """
        self.bins = bins

    def describe(self, image, mask=None):
        """
        Calculates the feature vector corresponding to the color histogram of the input image. Optionally, you can
        specify a masked region to focus on.

        :param image: Image whose RGB histogram descriptor will be computed.
        :param mask: (Optional) Mask to specify the region of interest to compute the RGB histogram.
        :return: Input image descriptive vector computer from the color histogram of the RGB channels.
        """
        print(self.bins)
        histogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)

        return histogram.flatten()