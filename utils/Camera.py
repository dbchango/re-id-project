import cv2

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read_video(self, extract_masks):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                r, output = extract_masks(frame)
                print(r["rois"])
                if r["rois"] != []:
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
                    # Cuadro raíz
                    cv2.rectangle(frame, start_point_r, end_point_r, color_r, thickness_r)
                yield r, frame
                # cv2.imshow("Output", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()