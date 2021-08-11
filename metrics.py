import numpy as np
from skimage.metrics import structural_similarity, hausdorff_distance

from utils import getrRect

def mean_std_compare(filtered_img, rRect=None):
    np_img = np.array(filtered_img)
    #np_img[rRect[0]:rRect[2], rRect[1]:rRect[3]]
    homog = getrRect(np_img, rRect)
    return (np.mean(homog), np.std(homog))


def NRMSE(gold_std, filtered_img):
    Igs = np.array(gold_std)
    If = np.array(filtered_img)
    return np.sqrt(np.sum((If-Igs)**2) / np.sum(Igs**2))


def SSIM(gold_std, filtered_img, kSize=9):
    assert kSize%2==1, "Tamanho da janela deve ser ímpar"
    Igs = np.array(gold_std).astype(np.uint8)
    If = np.array(filtered_img).astype(np.uint8)
    return structural_similarity(Igs, If, win_size=kSize)

def USDSAI(speckled_img, filtered_img, classes):
    limiar = classes.size * 0.005 # 5% do total dos pixels
    unique_classes = np.unique(classes)
    N = len(unique_classes)
    new_unique_classes = []
    for i in range(N):
        if np.sum(classes==unique_classes[i]) > limiar:
            new_unique_classes.append(unique_classes[i])
    unique_classes = np.array(new_unique_classes)
    N = len(unique_classes)

    means_sp = np.zeros((N,1))
    means_f = np.zeros((N,1))
    vars_sp = np.zeros(N)
    vars_f = np.zeros(N)

    for i in range(N):
        means_sp[i] = np.mean(speckled_img[classes==unique_classes[i]])
        means_f[i] = np.mean(filtered_img[classes==unique_classes[i]])
        vars_sp[i] = np.var(speckled_img[classes==unique_classes[i]])
        vars_f[i] = np.var(filtered_img[classes==unique_classes[i]])

    if (np.sum(vars_sp)==0):
        return 100000

    Q_sp = np.sum(np.tril(means_sp-means_sp.transpose())**2)/(N-1)/(np.sum(vars_sp))
    Q_f = np.sum(np.tril(means_f-means_f.transpose())**2)/(N-1)/(np.sum(vars_f))

    return Q_f/Q_sp
    

def HD(im1, im2):
    return hausdorff_distance(im1, im2)
