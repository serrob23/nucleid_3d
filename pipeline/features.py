import os, json, glob
import numpy as np
import copy
from scipy.spatial import Delaunay, Voronoi
from skimage.io import imread
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects
import pandas as pd


def getObjectProperties(labeled_image):

    """
    Returns labeled object properties in a pandas DataFrame for convienient
    sorting.


    Parameters
    ----------

    labled_image : 3D numpy array
        Segmented image of nuclei where each individual object has been
        assigned a unique integer idea.


    Returns
    -------

    object_props : pd.DataFrame
        DataFrame object with selected properties extracted using
        skimage.measure.regionprops_table



    """

    # object properties for extraction
    properties = ['equivalent_diameter', 'inertia_tensor',
                  'inertia_tensor_eigvals', 'major_axis_length',
                  'minor_axis_length', 'moments',
                  'moments_central', 'label', 'area',
                  'solidity', 'feret_diameter_max'
                  'moments_normalized', 'centroid', 'bbox',
                  'bbox_area', 'extent',
                  'convex_area', 'convex_image']

    # extract features and return as dataframe
    object_props = pd.DataFrame(regionprops_table(labeled_image,
                                                  properties=properties))
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


def fullworkflow(filename, min_size=150, return_objects=False):

    data = imread(filename)

    data = remove_small_objects(data, min_size=min_size)

    properties = getObjectProperties(data)

    centroids = getCentroids(properties)

    tesselD, tesselV = getTesselations(centroids['centroids'])

    graphs = {'delaunay': tesselD.__dict__,
              'voronoi': tesselV.__dict__}

    for gkey, gvalue in graphs.items():
        for key, value in gvalue.items():

            if type(value) == np.ndarray:
                graphs[gkey][key] = value.tolist()

    savedir = filename.split('.tif')[0] + '_features'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    prop_filename = os.path.join(savedir, 'region_properties.csv')
    json_filename = os.path.join(savedir, 'graphs.json')

    properties.to_csv(prop_filename)

    with open(json_filename, 'w') as f:
        thestring = json.dumps(graphs)
        json.dump(thestring, f, indent=4, sort_keys=True)
    f.close()

    if return_objects:
        return properties, graphs

    else:
        return


def loadPropTable(filepath):
    return pd.read_csv(filepath)
