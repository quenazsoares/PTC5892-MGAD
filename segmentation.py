from matplotlib import pyplot as plt
import numpy as np
from skimage import morphology
from skimage.segmentation import flood
from skimage.filters import threshold_multiotsu

import morphsnakes as ms



# https://scikit-image.org/docs/0.12.x/auto_examples/xx_applications/plot_coins_segmentation.html
def otsu(image, init_pixel, bias=1, **kwargs):
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image, classes=3)+bias

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    return flood(regions, init_pixel,)


# morphological_geodesic_active_contour
# Referência: https://github.com/pmneila/morphsnakes
def MGAC(img, init_ls, iterations=100, debug=False, **kwargs):
    """Gradient Vector Flow with morphological geodesic active contour.

    Args:
        img (numpy.ndarray): Image to be segmented.
        init_ls (numpy.ndarray): Initialization of the level-set.

    Returns:
        numpy.ndarray: Final segmentation (i.e., the final level set).
    """

    def visual_callback_2d(background, fig=None):
        """
        Returns a callback than can be passed as the argument `iter_callback`
        of `morphological_geodesic_active_contour` and
        `morphological_chan_vese` for visualizing the evolution
        of the levelsets. Only works for 2D images.
        Parameters
        ----------
        background : (M, N) array
            Image to be plotted as the background of the visual evolution.
        fig : matplotlib.figure.Figure
            Figure where results will be drawn. If not given, a new figure
            will be created.
        Returns
        -------
        callback : Python function
            A function that receives a levelset and updates the current plot
            accordingly. This can be passed as the `iter_callback` argument of
            `morphological_geodesic_active_contour` and
            `morphological_chan_vese`.
        """

        # Prepare the visual environment.
        if fig is None:
            fig = plt.figure()
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(background, cmap=plt.cm.gray)

        ax2 = fig.add_subplot(1, 2, 2)
        ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
        plt.pause(0.001)

        def callback(levelset):

            if ax1.collections:
                del ax1.collections[0]
            ax1.contour(levelset, [0.5], colors='r')
            ax_u.set_data(levelset)
            fig.canvas.draw()
            plt.pause(0.001)

        return callback

    gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=5.48)

    # Initialization of the level-set.

    # Callback for visual plotting
    if debug:
        callback = visual_callback_2d(img)
    else:
        callback =  lambda x: None

    # MorphGAC.
    return ms.morphological_geodesic_active_contour(gimg, iterations,
                                             init_level_set=init_ls,
                                             smoothing=1, threshold=0.31,
                                             balloon=-1, iter_callback=callback)
    

# Referência: # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html#sphx-glr-auto-examples-segmentation-plot-floodfill-py
def flood_fill(img, init_pixel, **kwargs):
    # flood function returns a mask of flooded pixels
    mask = flood(img, init_pixel, tolerance=30, connectivity=5)

    # Fill small holes with binary closing
    mask_postprocessed = morphology.binary_closing(
                    mask, morphology.disk(15))

    # Remove thin structures with binary opening
    mask_postprocessed_out = morphology.binary_opening(mask_postprocessed,
                                                morphology.disk(15))

    return mask_postprocessed_out


