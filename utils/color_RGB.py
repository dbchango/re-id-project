import cv2

class color_RGB:
    def grisColor(self, imggris):
        imgRGB = cv2.cvtColor(imggris, cv2.COLOR_BGR2RGB )
        return imgRGB