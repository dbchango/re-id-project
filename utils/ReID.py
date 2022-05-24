from utils.MaskRCNN import MaskRCNN
def extract_texture():

    return None # tensor

def compare_textur_info(item1, item2, threshold):
    return None

def extract_masks(frame):
    '''
    This function detect and segment pedestrians in an image and returns processed image
    :param frame:
    :return r, output:
    '''
    seg = MaskRCNN().load_model()
    r, output = seg.segmentFrame(frame)
    # funcion de pame -> salida: mask con las areas agregadas
    return r, output

