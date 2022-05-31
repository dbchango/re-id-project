import os
import cv2
def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        r, output = extract_masks(frame)
        yield r, output
