import os
import cv2
from utils.ReID import DPM
from utils.ReID import texture
def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        r, output = extract_masks(frame)
        if r["rois"] != []:
            x1,y1,x2,y2,cropc, cropt, cropp = DPM(r,frame)
            # print(start_point_rc, end_point_rc, start_point_rt, end_point_rt, start_point_rp, end_point_rp)
            #
            # Cuadro cabeza
            # cv2.rectangle(frame, start_point_rc, end_point_rc, (0, 0, 255), 2)
            #
            # # Cuadro torso
            # cv2.rectangle(frame, start_point_rt, end_point_rt, (0, 255, 0), 2)
            #
            # # Cuadro Pies
            # cv2.rectangle(frame, start_point_rp, end_point_rp, (0, 0, 0), 2)

            #Recortes

            #Cuadro raiz
            # cv2.imshow('Image', crop)

            # Cuadro cabeza
            cv2.imshow('Imagec', cropc)
            texture(cropc)

            # Cuadro torso
            cv2.imshow('Imaget', cropt)
            texture(cropt)

            # # Cuadro piernas
            cv2.imshow('Imagep', cropp)
            texture(cropp)

        yield r, output
