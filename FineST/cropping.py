import logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from matplotlib.path import Path
import numpy as np
from skimage import draw, measure, io
from PIL import Image
import scanpy as sc
Image.MAX_IMAGE_PIXELS = None

def create_mask(polygon, shape):
    """Create a mask for the given shape

    Parameters
    ----------
    polygon : (N, 2) array
        Points defining the shape.
    shape : tuple of two ints
        Shape of the output mask.

    Returns
    -------
    mask : (shape[0], shape[1]) array
        Boolean mask of the given shape.
    """
    polygon = polygon.iloc[:, -2:].values
    print("polygon: \n", polygon)
    polygon = np.clip(polygon, a_min=0, a_max=None)
    print("polygon adjusted: \n", polygon)
    ## Keep the order of x and y unchanged
    rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], shape) 
    # rr, cc = draw.polygon(polygon.iloc[:, -2], polygon.iloc[:, -1], shape) 

    ## Set negative coordinate to 0
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask, polygon



def crop_img_adata(roi_path, img_path, adata_path, crop_img_path, crop_adata_path, 
                   segment=False, save=None):
    """
    Crop an image and an AnnData object based on a region of interest.

    Parameters:
    roi_path : numpy.ndarray
        A numpy array specifying the region of interest.
    img_path : str
        The path to the image file.
    adata_path : str
        The path to the AnnData file.
    crop_img_path : str
        The path where the cropped image will be saved.
    crop_adata_path : str
        The path where the cropped AnnData object will be saved.
    save: bool, optional
        Whether to save the cropped image and AnnData object to files. 
        Default is None, which means not to save.

    Returns:
    tuple
        A tuple containing the cropped image and the cropped AnnData object.
    """

    roi_coords = pd.read_csv(roi_path)
    print("ROI coordinates from napari package: \n", roi_coords)

    img = io.imread(img_path)
    print("img shape: \n", img.shape)

    ## Create a mask for the region of interest
    mask, roi_coords = create_mask(roi_coords, img.shape[:2])
    ## Find the bounding box of the region of interest
    props = measure.regionprops_table(mask.astype(int), properties=('bbox',))

    minr = props['bbox-0'][0]
    minc = props['bbox-1'][0]
    maxr = props['bbox-2'][0]
    maxc = props['bbox-3'][0]

    cropped_img = img[minr:maxr, minc:maxc]
    print("cropped_img shape: \n", cropped_img.shape)

    if save:
        io.imsave(crop_img_path, cropped_img)

    adata = sc.read_h5ad(adata_path)

    print("The adata: \n", adata)
    print("The range of original adata: \n", 
          [[adata.obsm["spatial"][:,0].min(), adata.obsm["spatial"][:,0].max()], 
           [adata.obsm["spatial"][:,1].min(), adata.obsm["spatial"][:,1].max()]])
    
    ## replace x and y of adata
    roi_yx = roi_coords[:, [1, 0]]   
    # roi_yx = np.stack([roi_coords.iloc[:, -1], roi_coords.iloc[:, -2]]).T
    adata_roi = adata[Path(roi_yx).contains_points(adata.obsm["spatial"]), :].copy()

    ## Update the 'spatial' field of the AnnData object
    # print(roi_coords)
    # print([roi_coords[0][1], roi_coords[0][0]])

    ## Update the 'spatial' field of the AnnData object
    ## if you no need segment, then it can be omitted.
    if segment:
        if roi_coords[2][0] == 0: 
            adata_roi.obsm["spatial"] = adata_roi.obsm["spatial"] - \
                                        np.array([roi_coords[0][1], 0])
        else: 
            adata_roi.obsm["spatial"] = adata_roi.obsm["spatial"] - \
                                        np.array([roi_coords[0][1], roi_coords[0][0]])
    if save:
        adata_roi.write(crop_adata_path)

    return cropped_img, adata_roi