import numpy as np
from skimage.morphology import disk
import skimage.filters as ft
import torch
from skimage.exposure import equalize_adapthist
import warnings
from cellpose import models


warnings.filterwarnings(action='once')


def Pipeline(image_volume, do_3D=True, flow=0.0,
             diameter=17, stitch=0.1, net_avg=False,
             batch_size=6, use_CP=True, min_size=150):


    torch.cuda.empty_cache()

    if use_CP:
        model = models.Cellpose(gpu=True, model_type='nuclei', net_avg=net_avg)

    else:
        model = models.CellposeModel(gpu=True, pretrained_model='cyto',
                                     net_avg=net_avg)

    if do_3D:
        masks, flows, styles, diams = model.eval(image_volume,
                                                 batch_size=batch_size,
                                                 channels=[0, 0],
                                                 diameter=diameter,
                                                 do_3D=do_3D,
                                                 flow_threshold=flow,
                                                 min_size=min_size)

    else:
        image_list = np.split(image_volume, len(image_volume), axis=0)

        masks, flows, styles, diams = model.eval(image_list,
                                                 batch_size=batch_size,
                                                 channels=[0, 0],
                                                 diameter=diameter,
                                                 do_3D=do_3D,
                                                 flow_threshold=flow,
                                                 stitch_threshold=stitch,
                                                 normalize=False)
        masks = np.asarray(masks)

    return masks.astype(np.uint32)


def preProcessImage(image, radius=2):

    kernel = disk(radius)
    med_image = np.zeros(image.shape, dtype=image.dtype)

    for i, z in enumerate(image):
        med_image[i] = ft.rank.median(z, selem=kernel)

    med_image = (equalize_adapthist(med_image.astype(np.uint16))*image.max())

    return med_image.astype(np.uint16)


def runWorkflow(rootdir, filename, dirname, im_range = 500, z_start = 0, z_max = 4000):
    """
    Runs full pipeline workflow for nuclear segmentation
    
    
    Parameters
    ----------
    
    rootdir : str or pathlike
        Root directory for data
        
    
    filename : str
        Hdf5 file for processing. Assumed to have standard file structure with
    nuclear channel as ch00
    
    
    dirname : str
        Directory name for saving results. Will be within rootdir path, does not 
    need to exist.
    
    im_range : int, default 500
        First dimension image size to load into memory.
    
    
    """
    
    
    
    data_file = os.path.join(rootdir, filename)


    savedir = os.path.join(rootdir, dirname)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    f = h5py.File(data_file,'r')
    print(f['t00000/s00/0/cells'].shape, f['t00000/s00/0/cells'].dtype)

    for i in range(z_start,f['t00000/s00/0/cells'].shape[0],im_range):
        if i + im_range > z_max:
            return
        filename = os.path.join(savedir, 'region1_x_{:0>6d}_{:0>6d}_diam17.tif'.format(i,i+im_range))
        print(filename)
        image = f['t00000/s00/0/cells'][i:i+im_range].astype(np.uint16)
        image = preProcessImage(image)
        masks = Pipeline(image,diameter=17,do_3D=True, net_avg=True, batch_size = 6, flow=.4)
        sk.io.imsave(filename,masks)
        del masks
    return
