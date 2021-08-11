"""Planilha de análise com as colunas:

ID          : Identificador único do arquivo. MD5 do caminho do arquivo.
FILENAME    : Nome do arquivo.
GOLDSTD     : Nome do gold standard do arquivo.
SIMU_TYPE   : Nome da simulação utilizada para gerar o arquivo.
FILTER_NAME : Nome do filtro utilizado. 
ID_F        : Identificador único do arquivo com o filtro. MD5 do caminho do arquivo + nome do filtro
MEAN_HOMOG  : Média na região homogênea.
STD_HOMOG   : Desvio padrão na reião homogênea
NRMSE       : Erro quadrático médio.
SSIM        : Similaridade estrutural da imagem
USDSAI      :
SEGMENTATION: Nome da técnica de segmentação
ID_FS       : Identificador único do arquivo com o filtro e segmentação. MD5 do caminho do arquivo + nome do filtro + nome da segmentação.
ACCURACY    : Acurácia da segmentação.
F1_SCORE    : F1 score médio ponderado.
PRECISION   : Precição média ponderada.
RECALL      : Recall médio ponderado.

Segmentação só será aplicada para o gold standard "forms". Para os demais informar métricas como -1 e nome da segmentação como NONE.

Lembrar de não utilizar filter. Neste caso informar nome do filtro como NoFilter.

Utilizar métricas nas imagens gold standard. Neste caso não utilizar filtros e informar NONE; e informar tipo de simulação GOLDSTD.

"""

import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import time


from utils import load_img, save_img, md5
from filters import MGAD_Q, Med_circle, AWMF, AnisDiff, MGAD, Lee, Bi, ISF, ISFAD, GEO, NoFilter, MGAD_H
from metrics import NRMSE, mean_std_compare, SSIM, USDSAI, hausdorff_distance
from segmentation import otsu, MGAC, flood_fill

from skimage.segmentation import flood
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

result_filename = "Results.csv"
data_frame_columns = [
    "ID",
    "FILENAME",
    "GOLDSTD",
    "SIMU_TYPE",
    "FILTER_NAME",
    "ID_F",
    "FILTER_TIME", 
    "MEAN_HOMOG",
    "STD_HOMOG",
    "NRMSE",
    "SSIM",
    "USDSAI",
    "SEGMENTATION",
    "ID_FS",
    "ACCURACY",
    "F1_SCORE",
    "PRECISION",
    "RECALL", 
    "img_tp", 
    "img_fp",
    "img_ac",
    "HD_max",
    "HD_mean"
]

results = []


BASE_PATH = "simu\\simulacoes_organizado\\"
SIMU_NAMES = ["NONE", "MUST", "WeightedWhiteNoise", "Rayleight"]
GOLDSTD_NAMES = ["forms", "cistos", "checkboard3"]

FILTERS = [{"f": NoFilter, "name": "NoFilter"},
           {"f": AWMF, "name": "AWMF"},
           {"f": Med_circle, "name": "Median"},
           {"f": AnisDiff, "name": "AnisDiff"},
           {"f": MGAD, "name": "MGAD"},
           {"f": MGAD_Q, "name": "MGADB"},
           {"f": Lee, "name": "Lee"},
           {"f": Bi, "name": "Bilateral"},
           {"f": ISF, "name": "ISF"},
           {"f": ISFAD, "name": "ISFAD"},
           {"f": GEO, "name": "GEO"},
           {"f": MGAD_H, "name": "MGADd"}]

SEGMENTATIONS = [{"f": otsu, "name": "OTSU"},
                 {"f": MGAC, "name": "Morphological Snakes"},
                 {"f": flood_fill, "name": "Flood"}]

HOMOGENOUS_AREA = {"cistos": [40, 180, 520, 250], # Formato: [ax0_init, ax1_init, ax0_end, ax1_end]
                   "forms": [110, 10, 150, 150],
                   "checkboard3": [70, 70, 120, 120]}

FORMS_DATA = [{"name": "elipse",   "init_pixel":(70,120)},
              {"name": "trevo",    "init_pixel":(170,200)},
              {"name": "C",        "init_pixel":(310,225)},
              {"name": "retangulo","init_pixel":(220,90)}]

# SIMU_TYPE=SIMU_NAMES[0]
# GOLDSTD=GOLDSTD_NAMES[0]
for SIMU_TYPE in tqdm(SIMU_NAMES, desc="SIMU_TYPE", position=0):
    for GOLDSTD in tqdm(GOLDSTD_NAMES, desc="GOLDSTD", position=1, leave=False):
        files = glob(BASE_PATH + SIMU_TYPE + "\\" + GOLDSTD + "*")
        rRect = HOMOGENOUS_AREA[GOLDSTD]
        goldstd_filename = glob(BASE_PATH + "NONE\\" + GOLDSTD + "*")
        goldstd_img = load_img(goldstd_filename[0])
        goldstd_img_np = np.array(goldstd_img)

        # file=files[0]
        for file in tqdm(files, desc="FILES", position=2, leave=False):
            (folder, filename) = os.path.split(file)
            (filename, extension) = os.path.splitext(filename)
            ID = md5(file)
            folder_filtered = folder + "\\filtered\\"
            folder_segmented = folder + "\\segmented\\"
            if not os.path.exists(folder_filtered):
                os.makedirs(folder_filtered)
            if not os.path.exists(folder_segmented):
                os.makedirs(folder_segmented)

            speckled_img = load_img(file)
            speckled_img_np = np.array(speckled_img)
            # filter=FILTERS[0]
            for filter in tqdm(FILTERS, desc="FILTERS", position=3, leave=False):
                FILTER_NAME = filter["name"]
                filter_fcn = filter["f"]
                ID_F = md5(file+FILTER_NAME)
                start_time = time.time()
                filtered_img = filter_fcn(speckled_img, rRect=rRect)
                #filtered_img = MGAD(speckled_img, niter=100, k=1, beta=0.2, radius=2)

                FILTER_TIME = time.time() - start_time
                filtered_img_np = np.array(filtered_img)

                save_img(filtered_img, folder_filtered + filename + "_" + FILTER_NAME + ".png")

                # Métricas quantitativas
                (img_MEANH, img_STDH) = mean_std_compare(filtered_img, rRect)
                img_NRMSE = NRMSE(goldstd_img, filtered_img)
                img_SSIM = SSIM(goldstd_img, filtered_img)
                img_USDSAI = USDSAI(speckled_img_np, filtered_img_np, goldstd_img_np)

                # Segmentações
                if GOLDSTD == "forms":
                    # SEGMENTATION=SEGMENTATIONS[0]
                    for SEGMENTATION in tqdm(SEGMENTATIONS, desc="SEGMENT", position=4, leave=False):
                        SEGMENTATION_NAME = SEGMENTATION["name"]
                        segmentation_fcn = SEGMENTATION["f"]
                        ID_FS = md5(file+FILTER_NAME+SEGMENTATION_NAME)
                        
                        y_true = np.zeros(goldstd_img_np.shape)
                        y_pred = np.zeros(goldstd_img_np.shape)
                        hd =[]
                        class_id = 1
                        # FORM_DATA = FORMS_DATA[0]
                        for FORM_DATA in FORMS_DATA:
                            init_ls = np.array(load_img("simu\\initial_contour\\"+FORM_DATA["name"] + ".png"))
                            init_pixel = FORM_DATA["init_pixel"]
                            TRUE_LABELS = flood(goldstd_img, init_pixel)
                            y_true = y_true + class_id * TRUE_LABELS
                            PREDICTED_LABELS = segmentation_fcn(filtered_img_np, init_pixel=init_pixel, init_ls=init_ls)
                            y_pred = y_pred + class_id * PREDICTED_LABELS
                            hd.append(hausdorff_distance(TRUE_LABELS, PREDICTED_LABELS))
                            class_id += 1
                            
                        img_ACCURACY = accuracy_score(y_true.flat, y_pred.flat, )
                        #img_F1_SCORE = f1_score(y_true.flat, y_pred.flat, average='macro')
                        img_F1_SCORE = f1_score(y_true.flat, y_pred.flat, average='micro', labels=[1,2,3,4])
                        img_PRECISION = precision_score(y_true.flat, y_pred.flat, average='micro', labels=[1,2,3,4])
                        img_RECALL = recall_score(y_true.flat, y_pred.flat, average='micro', labels=[1,2,3,4])

                        img_tp = np.sum((y_true == y_pred) & (y_true > 0))/ np.sum(y_true > 0)
                        img_fp = np.abs(np.sum(y_pred > 0) - np.sum(y_true > 0))/np.sum(y_true > 0)
                        img_ac = (img_tp+(1-img_fp))/2
                        
                        hd = np.array(hd)
                        img_HD_max = np.max(hd)
                        img_HD_mean = np.mean(hd)

                        np.sum((y_pred > 0) & (y_true == 0))/ np.sum(y_true > 0)

                        results.append([ID, filename, GOLDSTD, SIMU_TYPE, FILTER_NAME, ID_F, FILTER_TIME, 
                            img_MEANH, img_STDH, img_NRMSE, img_SSIM, img_USDSAI, SEGMENTATION_NAME, 
                            ID_FS, img_ACCURACY, img_F1_SCORE, img_PRECISION, img_RECALL, img_tp, img_fp, 
                            img_ac, img_HD_max, img_HD_mean])

                        save_img(y_pred*50, folder_segmented + filename + "_" + FILTER_NAME + "_" + SEGMENTATION_NAME + ".png")
                else:
                    SEGMENTATION_NAME = "NONE"
                    ID_FS = md5(file+FILTER_NAME+SEGMENTATION_NAME)
                    img_ACCURACY = -1
                    img_F1_SCORE = -1
                    img_PRECISION = -1
                    img_RECALL = -1
                    img_tp = -1
                    img_fp = -1
                    img_ac = -1
                    img_HD_max = -1
                    img_HD_mean = -1
                    results.append([ID, filename, GOLDSTD, SIMU_TYPE, FILTER_NAME, ID_F, FILTER_TIME, 
                            img_MEANH, img_STDH, img_NRMSE, img_SSIM, img_USDSAI, SEGMENTATION_NAME, 
                            ID_FS, img_ACCURACY, img_F1_SCORE, img_PRECISION, img_RECALL, img_tp, img_fp,
                            img_ac, img_HD_max, img_HD_mean])
            
            final_results = pd.DataFrame(results, columns=data_frame_columns)
            final_results.to_csv(result_filename)


 