from utils.MaskRCNN import MaskRCNN
def extract_texture():

    return None # tensor

def compare_textur_info(item1, item2, threshold):
    return None

def extract_masks(frame):
    """
    This function detect and segment pedestrians in an image and returns processed image
    :param frame:
    :return results, processed_image:
    """
    seg = MaskRCNN().load_model()
    r, output = seg.segmentFrame(frame)

    # TODO: add area calculation function (@PAMELA), this function will calculate tha area of
    #  each person mask and add those to previous maskrcnn result

    return r, output

