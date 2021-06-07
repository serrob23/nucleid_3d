from ipywidgets import interact, fixed, IntSlider
import ipywidgets as widgets
import matplotlib.pyplot as plt

def plot_slice(im, axis, depth):
    '''Plots a specified slice of a 3D image.
    
    This function plots a slice of the 3D image array `im`. The direction and position of the slice
    is specified using `axis` and `depth`. 
    
    Parameters
    ----------
    im : array_like
        3D image array
    axis : int
        Axis along which to slice the 3D image. Takes values of 
        0, 1, or 2 corresponding to the x, y, and z axis. 
    depth : array_like
        The index of the slice along the given axes. 
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> plot_slice(im, 2, 100)
    
    Plots a slice of im along the z axis at index 100
    
    >>> plot_slice(im=ex_image, axis=1, depth=25)
    
    Plots a slice of ex_image along the y axis at index 25
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    
    if axis == 0:
        ax.imshow(im[depth, ...])
    elif axis == 1:
        ax.imshow(im[:, depth, :])
    elif axis == 2:
        ax.imshow(im[..., depth])
    else:
        raise ValueError('axis values must be 0, 1, or 2')
        return
    ax.axis('off')
    return
    
def plot_interact(im, axis=0):
    '''Displays an interactive plot of 3D image slices.
    
    This function plots slices of a 3D image array with slice direction specified by `axis`. 
    The plot includes an interactive slider where you can change the depth/index of 
    the slice along the given axis. 
    
    Parameters
    ----------
    im : array_like
        3D image array
    axis : int, optional
        Axis along which to slice the 3D image. Takes values of 
        0, 1, or 2 corresponding to the x, y, and z axis. 
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> plot_interact(im)
    
    Plots interactive graph of slices of im along x axis
    
    >>> plot_interact(ex_image, axis=2)
    
    Plots interactive graph of slices of ex_image along z axis
    '''
    try:
        im.shape[axis]
    except:
        raise ValueError('axis values must be 0, 1, or 2')
        return
    interact(plot_slice, 
             im=fixed(im),
             axis=fixed(axis),
             depth=widgets.IntSlider(min=0, max=im.shape[axis]-1, step=1, value=0))
    return
