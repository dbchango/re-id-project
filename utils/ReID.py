<<<<<<< HEAD
from utils.MaskRCNN import MaskRCNN
=======
#Importamos librerias
import cv2
from utils.MaskRCNN import MaskRCNN\
>>>>>>> d5e651b9cc7d20d82fbfcad30d8c969969889dc8

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt
import cv2

seg = MaskRCNN()

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

    r, output = seg.segment(frame)

    # TODO: add area calculation function (@PAMELA), this function will calculate tha area of
    #  each person mask and add those to previous maskrcnn result

    return r, output

def DPM(r,frame):
    #print(r["rois"])
    # Cuadro raiz
    x1 = r["rois"][0][0]
    y1 = r["rois"][0][1]
    x2 = r["rois"][0][2]
    y2 = r["rois"][0][3]

    # segmentación
    # Cuadro raíz
    # start_point_r = (y1, x1)
    # end_point_r = (y2, x2)
    # color_r = (255, 0, 0)
    # thickness_r = 2

    # Cuadro cabeza
    # xc = int(x1 + (x2 - x1) / 3)
    # start_point_rc = (y1, x1)
    # end_point_rc = (y2, xc)
    # color_rc = (0, 0, 255)
    # thickness_rc = 2

    # Cuadro torso
    # xt = int(x1 + (x2 - x1) * (2 / 3))
    # start_point_rt = (y1, x1 + xc)
    # end_point_rt = (y2, xt)
    # color_rt = (0, 255, 0)
    # thickness_rt = 2

    # Cuadro Pie
    # xp = int(x1 + (x2 - x1))
    # start_point_rp = (y1, xp)
    # end_point_rp = (y2, xt)
    # color_rp = (0, 0, 0)
    # thickness_rp = 2

    # # Cuadro raíz
    # cv2.rectangle(frame, start_point_r, end_point_r, color_r, thickness_r)
    #
    # # Cuadro cabeza
    # cv2.rectangle(frame, start_point_rc, end_point_rc, color_rc, thickness_rc)
    #
    # # Cuadro torso
    # cv2.rectangle(frame, start_point_rt, end_point_rt, color_rt, thickness_rt)
    #
    # # Cuadro Pies
    # cv2.rectangle(frame, start_point_rp, end_point_rp, color_rp, thickness_rp)

    # Cuadro raiz
    # y=x1
    # x=y1
    # h=x2
    # w=y2
    # crop = frame[y:y + h, x:x + w]

    # Cuadro cabeza
    xc1 = int(x1 + (x2 - x1) / 3)
    yc = x1
    xc = y1
    hc = xc1
    wc = y2
    cropc = frame[yc:yc + hc, xc:xc + wc]

    # Cuadro torso
    xt1 = int(x1 + (x2 - x1) * (2 / 3))
    yt = x1 + xc1
    xt = y1
    ht = xt1 - xc1
    wt = y2
    cropt = frame[yt:yt + ht, xt:xt + wt]

    # Cuadro piernas
    xp1 = int(x1 + (x2 - x1))
    yp = xt1
    xp = y1
    hp = xp1 - xt1
    wp = y2
    cropp = frame[yp:yp + hp, xp:xp + wp]

    return x1,y1,x2,y2,cropc,cropt,cropp

def texture(frame):
    # configuraciones de nuestro descriptor LBP
    radio = 3
    numpuntos = 8 * radio

    # Definimos una funcion que va a recibir la imagen
    def etiqueta(frame, lbp, labels):
        mask = np.logical_or.reduce([lbp == each for each in labels])
        return label2rgb(mask, frame, bg_label=0, alpha=0.5)

    def destacado(barras, indices):  # Aca vamos a pintar de color rojo esas barras destacadas del histograma
        for i in indices:
            barras[i].set_facecolor('r')  # Aca definimos el color

    # Leemos la imagen
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(img)  # Mostramos la imagen
    plt.show()

    # Ahora calculamos los LBP uniformes de la imagen
    METHOD = 'uniform'
    lbp = local_binary_pattern(img, numpuntos, radio, METHOD)

    def hist(ax, lbp):  # Ahora definimos una funcion para determinar el histograma
        n_conte = int(lbp.max() + 1)  # Definimos el valor maximo del LBP para visualizarlo
        return ax.hist(lbp.ravel(), density=True, bins=n_conte, range=(0, n_conte), facecolor='0.5')

    # Ploteamos los histogramas
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))  # definimos Finlas, columbnas y tamaño
    plt.gray()

    # Aca vamos a hacer la configuracion de cada grafico para resaltar diferentes texturas
    titles = ('Bordes', 'Planos', 'Esquinas')  # Asignamos titulos
    w = ancho = radio - 1
    etiqueta_bordes = range(numpuntos // 2 - w, numpuntos // 2 + w + 1)
    etiqueta_plano = list(range(0, w + 1)) + list(range(numpuntos - w, numpuntos + 2))
    i_14 = numpuntos // 4  # 1/4 del histograma
    i_34 = 3 * (numpuntos // 4)  # 3/4 del histograma
    etiqueta_esquinas = (list(range(i_14 - w, i_14 + w + 1)) + list(range(i_34 - w, i_34 + w + 1)))
    etiqueta_conj = (etiqueta_bordes, etiqueta_plano, etiqueta_esquinas)

    # Una vez configuremos los graficos vamos a mostrar la imagen resaltando diferentes tonalidades de la imagen
    for ax, labels, in zip(ax_img, etiqueta_conj):
        ax.imshow(etiqueta(img, lbp, labels))

    # Ahora agregamos los histogramas donde resaltamos las barras mas altas segun la tonalidad que resaltamos
    for ax, labels, name in zip(ax_hist, etiqueta_conj, titles):
        con, _, barras = hist(ax, lbp)
        destacado(barras, labels)
        ax.set_ylim(top=np.max(con[:-1]))  # Configuramos los limites
        ax.set_xlim(right=numpuntos + 2)
        ax.set_title(name)

    # Ahora quitamos los valores de los gráficos de la textura a analizar
    ax_hist[0].set_ylabel('Porcentaje')
    for ax in ax_img:
        ax.axis('off')
    plt.show()

    return plt.show()

