import os
import cv2
def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        r, output = extract_masks(frame)
        print(r["rois"])
        if r["rois"] != []:
            #Cuadro raiz
            x1 = r["rois"][0][0]
            y1 = r["rois"][0][1]
            x2 = r["rois"][0][2]
            y2 = r["rois"][0][3]

            # segmentación
            # Cuadro raíz
            start_point_r = (y1, x1)
            end_point_r = (y2, x2)
            color_r = (255, 0, 0)
            thickness_r = 2

            # Cuadro cabeza
            xc = int(x1 + (x2 - x1) / 3)
            start_point_rc = (y1, x1)
            end_point_rc = (y2, xc)
            color_rc = (0, 0, 255)
            thickness_rc = 2

            # Cuadro torso
            xt= int(x1+(x2-x1)*(2/3))
            start_point_rt = (y1, x1+xc)
            end_point_rt = (y2,xt)
            color_rt = (0, 255, 0)
            thickness_rt = 2

            # Cuadro Pie
            xp = int(x1 + (x2 - x1))
            start_point_rp = (y1, xp)
            end_point_rp = (y2, xt)
            color_rp = (0, 0, 0)
            thickness_rp = 2

            # Cuadro raíz
            cv2.rectangle(frame, start_point_r, end_point_r, color_r, thickness_r)

            # Cuadro cabeza
            cv2.rectangle(frame, start_point_rc, end_point_rc, color_rc, thickness_rc)

            # Cuadro torso
            cv2.rectangle(frame, start_point_rt, end_point_rt, color_rt, thickness_rt)

            # Cuadro Pies
            cv2.rectangle(frame, start_point_rp, end_point_rp, color_rp, thickness_rp)
        yield r, output
