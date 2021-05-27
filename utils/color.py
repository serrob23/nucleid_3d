import numpy as np
import matplotlib as mpl
import skimage as sk


def getColorVolume(image, alpha=None):

    colors = mpl.cm.plasma(np.arange(0, 255, 5), alpha=alpha)
    colored_image = (255*sk.color.label2rgb(image, bg_label=0,
                     colors=colors,
                     bg_color=[0, 0, 0])).astype(np.uint8)
    return colored_image
