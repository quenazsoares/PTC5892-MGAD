from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from utils import weighted_median, meanCircleFilter, varCircleFilter, create_circular_mask, getrRect
import numpy as np
import numpy.lib.stride_tricks
from cv2 import bilateralFilter, medianBlur, filter2D, GaussianBlur, Laplacian, CV_64F, CV_8U, resize

def NoFilter(img, **kwargs):
    return img

# Median Filter with Pillow library
def Med(img, radius, **kwargs):
    return img.filter(ImageFilter.MedianFilter(radius))


# Median Filter with OpenCV library
def Med_CV2(img, radius, **kwargs):
    return medianBlur(img, radius)


# Median Filter using circular kernel
def Med_circle(img, radius=5, PIL_flag=True, **kwargs):
    """Median filter with circular kernel.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        radius (int): Radius of the circular kernel.
        PIL_flag (bool, optional): Flag to return a Pillow.Image (True) or a numpy.ndarray (False). Defaults to False.

    Returns:
        Pillow.Image/numpy.ndarray: Filtered image.
    """
    kSize = 2*radius + 1  # Compute the kernel size
    # Get the image with a border of radius pixels
    img_pad = np.pad(img, radius, mode='edge')
    # Get a circular mask
    (mask, dist_from_center) = create_circular_mask(kSize, kSize)
    # Get a view of the image using a sliding window
    img_sliding = numpy.lib.stride_tricks.sliding_window_view(img_pad, (kSize, kSize))[:, :, mask]
    out_img = np.median(img_sliding, axis=-1)
    if PIL_flag:
        return Image.fromarray(out_img)
    return out_img


# Adaptative Weighted Median Filter
def AWMF(img, radius=5, c=3, base_weight=450, **kwargs):
    """Adaptative weighted median filter with circular kernel.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        radius (int): Radius of the circular kernel.
        c (float): Scaling constant.
        base_weight (int, optional): Weight of the central pixel. Defaults to 99.

    Returns:
        Pillow.Image: Filtered image.
    """
    (width, height) = img.size
    kSize = 2*radius + 1  # Compute the kernel size
    (mask, dist_from_center) = create_circular_mask(kSize, kSize)
    
    # Adiciona padding para que a imagem resultante seja de mesmo tamanho que a original
    img_pad = np.pad(img, radius, mode="edge")
    img_sliding = numpy.lib.stride_tricks.sliding_window_view(img_pad, (kSize, kSize))[:, :, mask]
    
    means = img_sliding.mean((2)).reshape(-1, 1)
    variances = img_sliding.var((2)).reshape(-1, 1)
    flat_img = img_sliding.reshape(height * width, -1)
    d = dist_from_center[mask].reshape(1, -1)
    means[means==0] = 0.1 # Evita que ocorra divisoes por zero na róxima linha

    weights = base_weight - c*d*variances/means
    out_img = np.empty((height * width), np.float64)
    
    weights = np.maximum(0, np.round(weights))

    for i in range(height*width):
        out_img[i] = weighted_median(flat_img[i], weights[i])
    return Image.fromarray(out_img.reshape(height, width))


# Anisotropic Diffusion Filter
def AnisDiff(img, niter=600, k=1, lambdaT=0.5, **kwargs):
    """Anisotropic diffusion filter. Implementation based on the article.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        niter (int, optional): Number of iterations. Defaults to 600.
        k (int, optional): Sensitivity to edges. Defaults to 1.
        lambdaT (float, optional): Time step of the diffusion.. Defaults to 0.5.

    Returns:
        Pillow.Image: Filtered image.
    """
    img = np.array(img)
    for i in range(niter):
        img_pad = np.pad(img, 1, constant_values=np.nan)
        d_img = np.array(
                [img_pad[2:, 1:-1]-img_pad[1:-1,1:-1],
                 img_pad[:-2, 1:-1]-img_pad[1:-1,1:-1],
                 img_pad[1:-1:, 2:]-img_pad[1:-1,1:-1],
                 img_pad[1:-1:, :-2]-img_pad[1:-1,1:-1]])
        c = 1/(1+(np.abs(d_img)/k)**2)
        ns = np.sum(1-np.isnan(c),axis=0)
        img += lambdaT*np.nansum(c*d_img, axis=0)/ns
    return Image.fromarray(img)
    

# Median Guided Anisotropic Diffusion Filter
def MGAD(img, niter=300, k=1, beta=0.2, radius=2, PIL_flag=True, **kwargs):
    """Median guided anisotropic filter.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        niter (int, optional): Number of iterations. Defaults to 600.
        k (int, optional): Sensitivity to edges. Defaults to 1.
        beta (float, optional): Influence of the median filter. Between 0 and 1. Defaults to 0.2.
        radius (int, optional): Radius of the circular kernel in median filter. Defaults to 2.

    Returns:
        Pillow.Image: Filtered image.
    """
    kSize = 2*radius + 1
    img = np.array(img)
    for i in range(niter):
        # Adiciona padding para que a imagem resultante seja de mesmo tamanho que a original
        img_pad = np.pad(img, 1, constant_values=np.nan)
        #f = Med_circle(img, kSize, False) # Obtém a imagem filtrada pela mediana
        f = Med_CV2(img.astype(np.uint8), kSize) 
        #f = Med_circle(img, radius, PIL_flag=False)
        # Adiciona padding para que a imagem resultante seja de mesmo tamanho que a original
        f_pad = np.pad(f, 1, constant_values=np.nan)
        # Obtém as derivadas direcionais E, W, N, S da imagem e da imagem filtrada pela mediana
        d_img = np.array(
                [img_pad[2:, 1:-1]-img_pad[1:-1,1:-1],
                 img_pad[:-2, 1:-1]-img_pad[1:-1,1:-1],
                 img_pad[1:-1:, 2:]-img_pad[1:-1,1:-1],
                 img_pad[1:-1:, :-2]-img_pad[1:-1,1:-1]])
        d_f = np.array(
                [f_pad[2:, 1:-1] - f_pad[1:-1,1:-1],
                 f_pad[:-2, 1:-1] - f_pad[1:-1,1:-1],
                 f_pad[1:-1:, 2:] - f_pad[1:-1,1:-1],
                 f_pad[1:-1:, :-2] - f_pad[1:-1,1:-1]])
        # Calcula o coeficiente de difusão
        c = 1/(1+(np.abs(d_f)/k)**2)
        # Calcula a imagem da próxima iteração
        img = (1-beta)*img + beta*f + np.nansum(c*d_img, axis=0)/4.0
        #print(i)
    if PIL_flag:
        return Image.fromarray(img)
    else:
        return img


# Lee Filter
def Lee(img, radius=10, rRect=None, **kwargs):
    """Lee Filter

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        radius (int): Radius of the circular kernel.
        rRect (list of int): Reference subregion. [start_dim1, end_dim1, start_dim2, end_dim2]

    Returns:
        Pillow.Image: Filtered image.
    """
    # Obtém o coeficiente de variação da região de referência.
    #reference = np.array(img)[rRect[0]:rRect[2], rRect[1]:rRect[3]]
    reference = getrRect(np.array(img), rRect)
    qH = np.std(reference)/np.mean(reference)
    # Obtém a imagem filtrada com media móvel
    mean_img = meanCircleFilter(img, radius)
    # Obtém o coeficiente de variação referente a cada pixel
    mean_img[mean_img==0] = 0.1 # Evita que ocorra divisão por zero na próxima linha
    q = varCircleFilter(img, radius, std_flag=True) / mean_img
    q[q==0] = 1e-12
    # Obtém o detector de borda
    alpha = np.clip(1 - (qH/q)**2, 0, 1)
    g = alpha*np.array(img) + (1-alpha)*mean_img
    return Image.fromarray(g)


# Bilateral Filter (using OpenCV library)
def Bi(img, sigmaD=5, sigmaR=50, **kwargs):
    return Image.fromarray(bilateralFilter(np.array(img), -1, sigmaR, sigmaD))


# Interference-based Speckle Filter
def ISF(img, radii=(15,3), **kwargs):
    """Interference-based speckle filter

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        radius (list of int): Radius of the circular kernel on the first and second median filter.

    Returns:
        Pillow.Image: Filtered image.
    """
    i_med = img.filter(ImageFilter.MedianFilter(radii[0]))
    i_c = Image.fromarray(np.maximum(img, i_med))
    return i_c.filter(ImageFilter.MedianFilter(radii[1]))


# Interference-based Speckle Filter followed by Anisotropic Difusion
def ISFAD(img, radii=(11,3), niter=500, k=1, lambdaT=0.5, **kwargs):
    IMG_ISF = ISF(img, radii)
    return AnisDiff(IMG_ISF, niter, k, lambdaT)


# Geometric Filter (gf4d)
def GEO(img, niter=15, **kwargs):
    def img_dir(img, dir):
        directions = [
            [(-1,0),((1,0))],
            [(-1,1),(1,-1)],
            [(0,1),(0,-1)],
            [(1,1), (-1,-1)]
        ]
        dir = directions[dir]
        a = np.roll(img, dir[0], (0,1))
        b = img.copy()
        c = np.roll(img, dir[1], (0,1))
        return (a,b,c)
    
    old_img = np.array(img)
    previous_mean = np.mean(old_img)
    for iter in range(niter):
        
        for dir in range(0,4): # Directions
            (a,b,c) = img_dir(old_img,dir)
            if dir in (0, 2):
                weight = 1
            else:
                weight = 1/(2**0.5)

            if iter %2 == 0:
                b += (a >= (b+2)) * weight
                b += ((a>b) & (b<=c)) * weight
                b += ((c>b) & (b<=a)) * weight
                b += (c >= (b+2)) * weight

                b -= (a >= (b-2)) * weight
                b -= ((a<b) & (b>=c)) * weight
                b -= ((c<b) & (b>=a)) * weight
                b -= (c >= (b-2)) * weight
            else:
                b -= (a >= (b-2)) * weight
                b -= ((a<b) & (b>=c)) * weight
                b -= ((c<b) & (b>=a)) * weight
                b -= (c >= (b-2)) * weight
                b += (a >= (b+2)) * weight
                b += ((a>b) & (b<=c)) * weight
                b += ((c>b) & (b<=a)) * weight
                b += (c >= (b+2)) * weight

            old_img = b.copy()
            
        #print(iter, np.mean(b)) # Use to debug the mean drift

    # Restaura a média, para evitar problemas com os pixels ficando negativo ou explodindo para além de 255 (8 bits)
    old_img = old_img - np.mean(old_img) + previous_mean
    return Image.fromarray(old_img)


# Stacked Median Filter with circular kernel and border enhance
def S_Med(img, radii=np.arange(1,6,2), PIL_flag=True, EN=False, **kwargs):
    """Stack Median filter with circular kernel.
    Returns the mean between a stack of median filtered images with different radius and bord enhancement.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        radii (list of int): Radii of the circular kernel.
        PIL_flag (bool, optional): Flag to return a Pillow.Image (True) or a numpy.ndarray (False). Defaults to False.

    Returns:
        Pillow.Image/numpy.ndarray: Filtered image.
    """
    kSize = 2*radii + 1  # Compute the kernel size
    np_img = np.array(img)
    stack_shape = (np_img.shape[0], np_img.shape[1], len(radii))
    stack_img = np.empty(shape=stack_shape)
    edge_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    for i,k in enumerate(kSize):
        if EN:
            temp = medianBlur(np_img.astype(np.uint8), k)
            edge = filter2D(temp, CV_64F, edge_kernel)
            temp = edge
        else:
            temp = medianBlur(np_img.astype(np.uint8), k)
            temp = temp.astype(np.float64)
        stack_img[:,:,i]=temp
    out_img = stack_img.mean(axis=2)
    if PIL_flag:
        return Image.fromarray(out_img)
    else:
        return out_img


# Median Guided Anisotropic Diffusion Filter with border enhancement
def MGAD_Q(img, niter=400, k=1, beta=0.15, radius=5, factor=50, radii=np.arange(1,6,2), **kwargs):
    """Median guided anisotropic filter with border enhancement.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        niter (int, optional): Number of iterations. Defaults to 600.
        k (int, optional): Sensitivity to edges. Defaults to 1.
        beta (float, optional): Influence of the median filter. Between 0 and 1. Defaults to 0.2.
        radius (int, optional): Radius of the circular kernel in median filter. Defaults to 2.
        factor (int, optional): Period of border enhancement.
        radii (list of int, optional): Radii used in stacked median filter

    Returns:
        Pillow.Image: Filtered image.
    """
    img = np.array(img)
    for i in range(niter):
        # Adiciona padding para que a imagem resultante seja de mesmo tamanho que a original
        img_pad = np.pad(img, 1, constant_values=np.nan)
        # Use the stacked median filter with border enhance each 'factor' times
        if (i+1) % factor==0:
            f = S_Med(Image.fromarray(img),EN=True, PIL_flag=False, radii=radii)
        else:
            f = Med_CV2(img.astype(np.uint8), radius) 
        
        # Adiciona padding para que a imagem resultante seja de mesmo tamanho que a original
        f_pad = np.pad(f, 1, constant_values=np.nan)
        # Obtém as derivadas direcionais E, W, N, S da imagem e da imagem filtrada pela mediana
        d_img = np.array(
                [img_pad[2:, 1:-1]-img_pad[1:-1,1:-1],
                 img_pad[:-2, 1:-1]-img_pad[1:-1,1:-1],
                 img_pad[1:-1:, 2:]-img_pad[1:-1,1:-1],
                 img_pad[1:-1:, :-2]-img_pad[1:-1,1:-1]])
        d_f = np.array(
                [f_pad[2:, 1:-1] - f_pad[1:-1,1:-1],
                 f_pad[:-2, 1:-1] - f_pad[1:-1,1:-1],
                 f_pad[1:-1:, 2:] - f_pad[1:-1,1:-1],
                 f_pad[1:-1:, :-2] - f_pad[1:-1,1:-1]])
        # Calcula o coeficiente de difusão
        c = 1/(1+(np.abs(d_f)/k)**2)
        # Calcula a imagem da próxima iteração
        img = (1-beta)*img + beta*f + np.nansum(c*d_img, axis=0)/4.0
        #print(i)
    return Image.fromarray(img)


# Median Guided Anisotropic Diffusion - Decimated Version
def MGAD_H(img, niter=300, k=1, beta=0.2, radius=2, **kwargs):
    """Median guided anisotropic filter - Decimated version.
    Decimates the image in four images that will be filtered by MGAD.
    Returns the mean between the four filtered images, resized to original size.

    Args:
        img (numpy.ndarray/Pillow.Image): Input image.
        niter (int, optional): Number of iterations. Defaults to 600.
        k (int, optional): Sensitivity to edges. Defaults to 1.
        beta (float, optional): Influence of the median filter. Between 0 and 1. Defaults to 0.2.
        radius (int, optional): Radius of the circular kernel in median filter. Defaults to 2.

    Returns:
        Pillow.Image: Filtered image.
    """

    img_np = np.array(img)
    # Decima a imagem em quatro
    img_Q1 = img_np[0::2, 0::2]
    img_Q2 = img_np[1::2, 0::2]
    img_Q3 = img_np[1::2, 1::2]
    img_Q4 = img_np[0::2, 1::2]

    # Filtra as quatro imagens
    img_Q1_f = MGAD(img_Q1, niter=int(niter/2), k=k, beta=beta, radius=int(radius/2), PIL_flag=False)
    img_Q2_f = MGAD(img_Q2, niter=int(niter/2), k=k, beta=beta, radius=int(radius/2), PIL_flag=False)
    img_Q3_f = MGAD(img_Q3, niter=int(niter/2), k=k, beta=beta, radius=int(radius/2), PIL_flag=False)
    img_Q4_f = MGAD(img_Q4, niter=int(niter/2), k=k, beta=beta, radius=int(radius/2), PIL_flag=False)

    # Redimensiona as imagens para o tamanho original
    img_Q1_f2 = resize(img_Q1_f, (img_np.shape[1], img_np.shape[0]))
    img_Q2_f2 = resize(img_Q2_f, (img_np.shape[1], img_np.shape[0]))
    img_Q3_f2 = resize(img_Q3_f, (img_np.shape[1], img_np.shape[0]))
    img_Q4_f2 = resize(img_Q4_f, (img_np.shape[1], img_np.shape[0]))

    # Obtém a média entre as quatro imagens filtradas
    img_out = (img_Q1_f2 + img_Q2_f2 + img_Q3_f2 + img_Q4_f2)/4

    return Image.fromarray(img_out)
