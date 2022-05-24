from utils.Model import MaskRCNN
def extract_texture():

    return None # tensor

def compare_textur_info(item1, item2, threshold):
    return None

def extract_masks(images):
    model = MaskRCNN(batch_size=len(images)).loadmodel()
    mask_rcnn_results = model.detect(images, verbose=0)
    # funcion de pame -> salida: mask con las areas agregadas
    return mask_rcnn_results

