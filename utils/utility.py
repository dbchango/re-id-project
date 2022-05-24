import os

from Mask_RCNN.mrcnn import visualize
import numpy as np
import cv2

CLASS_NAMES_MASKRCNN = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']

def get_color_dict():
  all_colors = visualize.random_colors(100)
  color_dict = {}
  i = 0
  for c in CLASS_NAMES_MASKRCNN:
    if not c in color_dict:
      color_dict[c] = all_colors[i]
      i = i+1
  color_dict["background"] = (0,0,0)
  return color_dict

COLOR_MAP = get_color_dict()


def label_to_color_image(labels):
  # Adds color defined by the dataset colormap to the label.
  h, w = labels.shape
  img = np.zeros([h, w, 3])
  img = np.zeros((h, w), dtype=(float, 3))
  for i in range(h):
    for j in range(w):
      img[i][j] = np.array(COLOR_MAP[CLASS_NAMES_MASKRCNN[labels[i][j]]])
  img = img * 255
  return img.astype(np.uint8)


def combine_masks(img, result):
  boxes = result['rois']
  masks = result['masks']
  class_ids = result['class_ids']

  N = boxes.shape[0]
  h, w, c = img.shape
  seg_map = np.zeros((h, w))
  for i in range(N):
    mask = masks[:, :, i]
    mask = mask.astype(np.uint8)
    seg_map = seg_map + mask * class_ids[i]

  return seg_map.astype(np.uint8)


def merge_images(foreground, background, alpha=0.3):
  out_img = np.zeros(background.shape, dtype=background.dtype)
  out_img[:, :, :] = (alpha * background[:, :, :]) + ((1 - alpha) * foreground[:, :, :])
  return out_img


def get_masked_image(image, result):
  """
  Applies masks from the results to the given image

  """
  boxes = result['rois']
  masks = result['masks']

  N = boxes.shape[0]
  if not N:
    print("\n*** No instances to display *** \n")

  colors = visualize.random_colors(N)
  masked_image = image.astype(np.uint32).copy()

  for i in range(N):
    color = colors[i]

    # Mask
    mask = masks[:, :, i]
    masked_image = visualize.apply_mask(masked_image, mask, color)
  return masked_image.astype(np.uint8)


def print_fps(video):
  # Find OpenCV version
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

  if int(major_ver) < 3:
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  else:
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
  """
  Create a video from a list of images.

  @param      outvid      output video
  @param      images      list of images to use in the video
  @param      fps         frame per second
  @param      size        size of each frame
  @param      is_color    color
  @param      format      see http://www.fourcc.org/codecs.php
  @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

  The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
  By default, the video will have the size of the first image.
  It will resize every image to this size before adding them to the video.
  """
  from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
  fourcc = VideoWriter_fourcc(*format)
  vid = None
  for image in images:
    if not os.path.exists(image):
      raise FileNotFoundError(image)
    img = imread(image)
    if vid is None:
      if size is None:
        size = img.shape[1], img.shape[0]
      vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
    if size[0] != img.shape[1] and size[1] != img.shape[0]:
      img = resize(img, size)
    vid.write(img)
  vid.release()
  return vid