#Importamos librerias
import pandas as pd
import os
from pathlib import Path
from utils.MaskRCNN import MaskRCNN
import numpy as np
import cv2
from utils.ReID import mask_area_perimeter

seg = MaskRCNN()

def create_dataset_area_mask(input_images_path):
  files_names= os.listdir(input_images_path)
  dataset_list = []
  #Create a DataFrame object
  df = pd.DataFrame(dataset_list, columns = ['area' , 'perimeter', 'person_class_name'])
  for file_name in files_names:
    image_path = input_images_path+"/"+file_name
    imagen1 = cv2.imread(image_path)
    segmentacion, output  = seg.segment(imagen1)
    #segmentacion=segmentacion[0]
    silhouette=segmentacion['masks'][:,:,0]
    test = np.array(silhouette, dtype='uint8')
    area, perimeter = mask_area_perimeter(test)
    person_class_name = Path(image_path).stem
    df = df.append({'area': area, 'perimeter': perimeter, 'person_class_name': person_class_name}, ignore_index=True)

  return df


if __name__ == '__main__':
    dataset_mask = create_dataset_area_mask("./Datasets/videos/cam1/pame")
    dataset_mask.to_csv("dataset_characteristics_mask.csv")
    print(dataset_mask)
