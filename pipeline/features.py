import os, json, glob
import numpy as np
import copy
from scipy.spatial import Delaunay, Voronoi
import scipy.ndimage as ndi
import scipy.stats as stat
from skimage.io import imread
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, regionprops_table, shannon_entropy, marching_cubes, mesh_surface_area
from skimage.morphology import remove_small_objects
import pandas as pd
import h5py as h5


def std_intensity(regionmask, intensity):
    """
    Returns  standard deviation of intensity region
    
    
    Parameters
    ----------
    
    regionmask : ndarray
        Labeled mask for object region, required by extra_properties
        in regionprops_table.
    
    intensity : ndarray
        corresponding intensity image for masked region
        
    
    Returns
    -------
    
    std : float
        standard deviation via numpy
    
    """

    return np.std(intensity)

def var_intensity(regionmask, intensity):
    """
    Returns variance of intensity region
    
    
    Parameters
    ----------
    
    regionmask : ndarray
        Labeled mask for object region, required by extra_properties
        in regionprops_table.
    
    intensity : ndarray
        corresponding intensity image for masked region
        
    
    Returns
    -------
    
    var : float
        variance via numpy
    
    """
    return np.var(intensity)

def entropy_intensity(regionmask, intensity):
    """
    Returns shannon entropy of intensity region
    
    
    Parameters
    ----------
    
    regionmask : ndarray
        Labeled mask for object region, required by extra_properties
        in regionprops_table.
    
    intensity : ndarray
        corresponding intensity image for masked region
        
    
    Returns
    -------
    
    entropy : float
        shannon entropy via skimage.measure
    """
    
    return shannon_entropy(intensity)

def surface_area(regionmask):
    
    #pad with zeros to prevent holes at edges
    regionmask = np.pad(regionmask, [(1,1),(1,1),(1,1)], 'constant')
    
    #fill holes to generate solid object
    structure = np.ones((3,) * regionmask.ndim)
    filled_image = ndi.binary_fill_holes(regionmask, structure)
    
    #get mesh surface of object
    verts, faces, normals, values = marching_cubes(filled_image, mask=filled_image)
    
    #return mesh surface area
    return mesh_surface_area(verts, faces)
    

def getObjectProperties(labeled_image, intensity_image):
    
    """
    Returns labeled object properties in a pandas DataFrame for convienient sorting.
    
    
    Parameters 
    ----------
    
    labled_image : 3D numpy array
        Segmented image of nuclei where each individual object has been assigned a 
        unique integer idea.
        
        
    intensity_image : 3D numpy array
        3D intensity image of nuclei, assumed np.uint16.
        
    
    Returns
    -------
    
    object_props : pd.DataFrame
        DataFrame object with selected properties extracted using skimage.measure.regionprops_table
    
    
    
    """
    
    #object properties for extraction
    properties=[ 'equivalent_diameter', 'inertia_tensor_eigvals', 
                'major_axis_length', 'minor_axis_length', 
                'label', 'area',
                'solidity', 'feret_diameter_max', 
                'centroid', 'bbox', 
                'bbox_area', 'extent',
                'convex_area', 'min_intensity',
                'max_intensity', 'mean_intensity',
                'extent','image', 'intensity_image']
    
    #extract features and return as dataframe
    object_props = pd.DataFrame(regionprops_table(labeled_image, 
                                                  intensity_image=intensity_image, 
                                                  properties=properties,
                                                  extra_properties = (std_intensity, 
                                                                      var_intensity,
                                                                      entropy_intensity,
                                                                      surface_area)))
    
    object_props = object_props[object_props.columns.difference(['image', 
                                                                'intensity_image'])]
    
    return object_props


def getCentroids(proptable):

    """
    Returns labeled object centroids and labels in a dictionary.


    Parameters
    ----------

    proptable : pd.DataFrame
        labeled object properties with centroid & label columns


    Returns
    -------

    props_dict : dict
        Dictionary with 'centroids' and 'labels' as keys, with corresponding
    centroids and labels extracted from proptable as numpy arrays.
    """

    props_dict = {}

    # get centroid column titles
    filter_col = [col for col in proptable if col.startswith('centroid')]

    props_dict['centroids'] = proptable[filter_col].to_numpy().astype(int)
    props_dict['labels'] = proptable['label'].to_numpy()

    return props_dict


def getTesselations(centroids):

    """
    Return two graph based features from the scipy.spatial module


    Parameters
    ----------

    centroids : numpy array
        Array of centroids extracted from segmented nuclei


    Returns
    -------

    tesselD : scipy.spatial.Delaunay
        Fully connected graph based feature where nuclear centroids are
    input as nodes on the graph.

    tesselV : scipy.spatial.Voronoi
        Region based graph (derived from Delaunay) where individual regions
    are grown from points i.e nuclear centroids.

    """

    # extract delaunay diagram from scipy.spatial
    tesselD = Delaunay(centroids)

    # extract voronoi diagram from scipy.spatial
    tesselV = Voronoi(centroids)

    return tesselD, tesselV


def cropImage(image, image_props, object_label, clean=False):
    """
    crops section of input image based on bounding box of labeled objects

    labeled objects are determined by the object_label which is a label in a
    property table


    Parameters
    ----------

    image : 3D numpy array
        labeled segmented image of nuclei

    image_props : pd.DataFrame
        pandas dataframe of properties with label and bbox as extracted
    features

    object_label : int
        label of object to crop from input image

    clean : bool, optional
        clear objects without input label


    Returns
    -------

    crop : 3D numpy array
        cropped region containing the labeled object, crop coordinates are
    based on the bounding box.

    """

    assert(type(object_label) == int)

    prop = image_props.loc[image_props['label'] == object_label]

    if len(image.shape) == 2:
        coords = [prop['bbox-0'].values[0], prop['bbox-2'].values[0], 
                  prop['bbox-1'].values[0], prop['bbox-3'].values[0]]
        print(coords)

        crop = copy.deepcopy(image[coords[0]:coords[1], coords[2]:coords[3]])

    else:
        coords = [prop['bbox-0'].values[0], prop['bbox-3'].values[0],
                  prop['bbox-1'].values[0], prop['bbox-4'].values[0],
                  prop['bbox-2'].values[0], prop['bbox-5'].values[0]]
        crop = copy.deepcopy(image[coords[0]:coords[1],
                             coords[2]:coords[3],
                             coords[4]:coords[5]])

    if clean:
        crop = np.ma.masked_where(crop != object_label, crop).filled(0)
        crop = (crop > 0).astype(int)
    return crop

def fullworkflow(mask_filename,
                 h5_filename,
                 z_range,
                 xy_range=None,
                 min_size=150):
    
    """
    Run full feature extraction workfloww on segmented masks and 
    3D intensity data. Saves features to csv file.
    
    
    Parameters
    ----------
    
    mask_filename : str or pathlike
        Path to segmented mask file, assumed tif.
        
    h5_filename : str or pathlike
        Path to intensity image  file, assumed hdf5.
        
    z_range : tuple
        Tuple of ints for z range to read intensity data from
    h5 file.
    
    xy_range : None or tuple
        Tuple for xy coordinates to read intenstity data from
    a particular section of h5 file in the other two dimmensions. 
    
    min_size : int
        Default is 150. Minimum object size for masks, used for
    remove_small_objects from skimage.

    Returns
    -------
    
    
    """
    
    print('Reading masks from:', mask_filename)
    mask = imread(mask_filename)

    print('Removing small objects from mask')
    mask = remove_small_objects(mask, min_size=min_size)
    
    print('Mask datatype:', mask.dtype)
    
    print('Reading intensity data from: \n %s, \n z-range %s'%(h5_filename, z_range))
    f = h5.File(h5_filename,'r')
    
    if xy_range is None:
        intensity_image = f['t00000/s00/0/cells'][z_range[0]:z_range[1]]
    
    else: 
        x_range = xy_range[0]
        y_range = xy_range[1]
        intensity_image = f['t00000/s00/0/cells'][z_range[0]:z_range[1],
                                                  x_range[0]:x_range[1],
                                                  y_range[0]:y_range[1]]
    f.close()
    
    print('Exctracting features')
    properties = getObjectProperties(mask, intensity_image)
    
    properties = getPropertyDescriptors(properties)
    

    savedir = mask_filename.split('.tif')[0] + '_features'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    print('Saving data')
    prop_filename = os.path.join(savedir, 'region_properties.csv')

    properties.to_csv(prop_filename)

    return


def getPropertyDescriptors(properties):
    """
    Gets property descriptors for a set of object properties: (volume, inertia tensor eigenvalue 0, 
    entropy intensity, convex_area, major axis length, minor axis length, surface area, solidity, extent)
    for each property it will get: min/max, mean, variance, skewness, kurtosis. 


    Parameters
    ----------

    properties : pandas dataframe
        Pandas dataframe generated from getObjectProperties
    
    """

    def describeProperty(properties, key):
        """
        Gets statistical descriptors from input data.

        Parameters
        ----------

        properties : pandas dataframe
            Pandas dataframe generated from getObjectProperties
        
        key : str
            Column name in properties dataframe.

        
        Returns
        -------

        propdict : dict
            Dictionary with key/value entries corresponding to statistical descriptors.
        
        """
        propdict = {}
        
        propdict['nobs'], propdict['minmax'], propdict['mean'], \
        propdict['var'], propdict['skew'], propdict['kurt'] = stat.describe(properties[key])
        propdict['IQR'] = stat.iqr(properties[key])
        return propdict
    
    #volume descriptors
    vol_props  = describeProperty(properties, 'area')
    properties['vol_minmax'] = vol_props['minmax']
    properties['vol_mean'] = vol_props['mean']
    properties['vol_var'] = vol_props['var']
    properties['vol_skew'] = vol_props['skew']
    properties['vol_kurt'] = vol_props['kurt']
    properties['vol_IQR'] = vol_props['IQR']
    
    #inertia tensor 0 descriptors
    inertiaeig_props = describeProperty(properties, 'inertia_tensor_eigvals-0')
    properties['inertia_tensor_0_minmax'] = inertiaeig_props['minmax']
    properties['inertia_tensor_0_mean'] = inertiaeig_props['mean']
    properties['inertia_tensor_0_var'] = inertiaeig_props['var']
    properties['inertia_tensor_0_skew'] = inertiaeig_props['skew']
    properties['inertia_tensor_0_kurt'] = inertiaeig_props['kurt']
    properties['inertia_tensor_0_IQR'] = inertiaeig_props['IQR']
    
    #entropy descriptors
    entropy_props = describeProperty(properties, 'entropy_intensity')
    properties['entropy_minmax'] = entropy_props['minmax']
    properties['entropy_mean'] = entropy_props['mean']
    properties['entropy_var'] = entropy_props['var']
    properties['entropy_skew'] = entropy_props['skew']
    properties['entropy_kurt'] = entropy_props['kurt']
    properties['entropy_IQR'] = entropy_props['IQR']
    
    #convex area descriptors
    convex_props = describeProperty(properties, 'convex_area')
    properties['convex_minmax'] = convex_props['minmax']
    properties['convex_mean'] = convex_props['mean']
    properties['convex_var'] = convex_props['var']
    properties['convex_skew'] = convex_props['skew']
    properties['convex_kurt'] = convex_props['kurt']
    properties['convex_IQR'] = convex_props['IQR']
    
    #major axis length descriptors
    major_props = describeProperty(properties, 'major_axis_length')
    properties['major_axis_minmax'] = major_props['minmax']
    properties['major_axis_mean'] = major_props['mean']
    properties['major_axis_var'] = major_props['var']
    properties['major_axis_skew'] = major_props['skew']
    properties['major_axis_kurt'] = major_props['kurt']
    properties['major_axis_IQR'] = major_props['IQR']
    
    #minor axis length descriptors
    minor_props = describeProperty(properties, 'minor_axis_length')
    properties['minor_axis_minmax'] = minor_props['minmax']
    properties['minor_axis_mean'] = minor_props['mean']
    properties['minor_axis_var'] = minor_props['var']
    properties['minor_axis_skew'] = minor_props['skew']
    properties['minor_axis_kurt'] = minor_props['kurt']
    properties['minor_axis_IQR'] = minor_props['IQR']
    
    #surface area descriptors
    SA_props = describeProperty(properties, 'surface_area')
    properties['surface_area_minmax'] = SA_props['minmax']
    properties['surface_area_mean'] = SA_props['mean']
    properties['surface_area_var'] = SA_props['var']
    properties['surface_area_skew'] = SA_props['skew']
    properties['surface_area_kurt'] = SA_props['kurt']
    properties['surface_area_IQR'] = 
    
    #solidity descriptors
    solidity_props = describeProperty(properties, 'solidity')
    properties['solidity_minmax'] = solidity_props['minmax']
    properties['solidity_mean'] = solidity_props['mean']
    properties['solidity_var'] = solidity_props['var']
    properties['solidity_skew'] = solidity_props['skew']
    properties['solidity_kurt'] = solidity_props['kurt']
    properties['solidity_IQR'] = solidity_props['IQR']
    
    #extent descriptors
    extent_props = describeProperty(properties, 'extent')
    properties['extent_minmax'] = extent_props['minmax']
    properties['extent_mean'] = extent_props['mean']
    properties['extent_var'] = extent_props['var']
    properties['extent_skew'] = extent_props['skew']
    properties['extent_kurt'] = extent_props['kurt']
    properties['extent_IQR'] = extent_props['IQR']
    
    
    return properties
    
    
def sortedWorkflow(rootdir, 
                   segmentation_dir, 
                   file_title=None,
                   xy_range=None
                  ):
    """
    Runs feature extraction workflow by sorting mask files in root directory.
    
    
    Parameters
    ----------
    
    rootdir : str or pathlike 
        Root directory where 3D intensity data (assumed HDF5) and segmented mask directory
    is stored.
    
    segmentation_dir : str or pathlike
        Directory where segmented masks are stored. Assumed hierarchy is that segmentation_dir is
    within rootdir.
    
    file_title : str or None (optional)
        Default is none. File title for hdf5 file to read segmented data.
        
    
    Returns
    -------
    
    
    """
    
    
    
    if file_title is None:
        h5_file = glob.glob(rootdir + os.sep + '*.h5')
    
    else:
        h5_file = glob.glob(rootdir + os.sep + '*%s*.h5'%(file_title))
        print(h5_file)
    
    segmented_files = sorted(glob.glob(rootdir + os.sep + segmentation_dir + os.sep + '*.tif'))
    
    
    for file in segmented_files:
        split_filename = file.split('_')
        z_range = (int(split_filename[-3]),int(split_filename[-2]))
        print(z_range)
        
        fullworkflow(file, h5_file[0], z_range, xy_range)
    
    compileProps(rootdir, 'complete_props.csv')


def loadPropTable(filepath):
    return pd.read_csv(filepath)
