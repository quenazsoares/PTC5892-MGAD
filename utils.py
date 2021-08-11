import numpy as np
import cv2
from PIL import Image
import hashlib

# Mediana ponderada de um vetor
def weighted_median(data, weights):
    """ Find the wheighted median of an array, based on a cumulative 
    sum of the weights sorted by the data.

    Args:
        data (numpy.ndarray): Array of data to find the median.
        weights (numpy.ndarray): Weights of each data.

    Returns:
        float: Weighted median of the data array.
    """
    # Assures that data and weights are 1-dimensional arrays
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    assert data.ndim==1, "DATA array must be 1-dimensional"
    assert weights.ndim==1, "WEIGHTS array must be 1-dimensional"

    index = np.argsort(data)   # Get the index of sorted data array
    s_data = data[index]       # Get the sorted data array
    s_weights = weights[index] # Get weights array sorted by the data array
    midpoint = 0.5 * np.sum(s_weights) # Finds the midpoint
    # Verify the condition of a weight greater than the midpoint, 
    # scenario where the corresponding data is the median.
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)  # Calculte the cumulative sum of the weights
        # Find the index related to the midpoint
        idx = np.where(cs_weights <= midpoint)[0][-1]
        # Check the scenario where the median is between two values
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

#Filtro de média com kernel circular
def meanCircleFilter(img, radius, PIL_flag=False):
    """ Moving windows average filter with circular window.

    Args:
        img (Pillow.Image/numpy.ndarray): Image to be filtered
        radius (int): Radius of the circular window.
        PIL_flag (bool, optional): Flag to return a Pillow.Image (True) or a numpy.ndarray (False). Defaults to False.

    Returns:
        Pillow.Image/numpy.ndarray: Filtered image.
    """
    # Compute the kernel size
    kSize = 2*radius + 1
    # Get a circular kernel with the specified radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kSize, kSize)).astype(float)
    kernel /= np.sum(kernel) # Ensure the kernel has a unitary sum
    np_img = np.array(img)   # Convert the image to numpy.ndarray
    # Apply the kernel along the input image
    out = cv2.filter2D(np_img, -1, kernel)
    # Return the output image as ndarray or Image, according to the flag
    if PIL_flag:
        return Image.fromarray(out)
    else:
        return out

# Filtro de Variancia
def varCircleFilter(img, radius, std_flag=False, PIL_flag=False):
    """ Moving window variance filter with circular window.

    Args:
        img (Pillow.Image/numpy.ndarray): Image to be filtered
        radius (int): Radius of the circular window.
        std_flag (bool, optional): Flag to return the standard deviation instead of the variance. Defaults to False.
        PIL_flag (bool, optional): Flag to return a Pillow.Image (True) or a numpy.ndarray (False). Defaults to False.

    Returns:
        Pillow.Image/numpy.ndarray: Filtered image.
    """
    np_img = np.array(img)  # Convert the image to numpy.ndarray
    # Computes variance using the squared mean image and the mean of the squared image.
    var = (np.array(meanCircleFilter(np_img**2, radius)) - 
            np.array(meanCircleFilter(np_img, radius)) **2)
    # Check if the std will be outputed
    if std_flag:
        out = np.abs(var) ** 0.5
    else:
        out = var
    # Return the output image as ndarray or Image, according to the flag
    if PIL_flag:
        return Image.fromarray(out)
    else:
        return out

# Gera uma mascara circular e as distâncias até o centro
def create_circular_mask(h, w, center=None, radius=None):
    """ Generate circular mask and distances to the center.

    Args:
        h (int): Height of the circular mask.
        w (int): Width of the circular mask.
        center ([int, int], optional): Center of the mask. If None, the center will be in [h/2, w/2]. Defaults to None.
        radius (int, optional): Radius. If None, the radius will be the smallest distance between the center and the limits. Defaults to None.

    Returns:
        (numpy.ndarray, numpy.ndarray): Mask in boolean ndarray and cartesian distances from the center in float ndarray.
    """
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return (mask, dist_from_center)

# Carrega as imagens independente da resolução de intensidade
def load_img(fp, dtype=np.float64):
    return Image.fromarray(
        np.array(Image.open(fp)).astype(dtype),
        mode=None
        )


def save_img(img, fp):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(fp)


def md5(str):
    return hashlib.md5("str".encode('utf-8')).hexdigest()


def getrRect(mat, rRect):
    return mat[rRect[0]:rRect[2], rRect[1]:rRect[3]]