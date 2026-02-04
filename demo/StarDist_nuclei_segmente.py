## 2025.05.20 LLY adjust the code, part of it is from SpatialScopeNS.py
## 2026.02.03 LLY add the annotation and clean code, rename  SpatialScopeNS to FineST_StarDist_nuclei


import scanpy as sc
import squidpy as sq
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import argparse
import anndata
import pandas as pd
from PIL import Image 
Image.MAX_IMAGE_PIXELS = None
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FineST.utils import *  # need install FineST package if put in /demo file not depend FineST file
from matplotlib.path import Path
import numpy as np
from skimage import draw, measure, io


class FineST_StarDist_nuclei:
    """
    FineST StarDist Nuclei Segmentation class for spatial transcriptomics data.
    - Performs nuclei segmentation on H&E stained tissue images using the StarDist model. 
    - Integrates ST data coordinates with image data, allowing for single-nuclei resolution.
    
    The class supports two modes:
    1. Full mode: Processes the whole slide image (WSI) and ST data
    2. ROI mode: Crops image with ST data of region of interest (ROI) by coordinates from Napari.
    
    Main Workflow:
    1. Load ST data (AnnData format) and H&E stained image.
    2. [Optionally] Crop image and ST data based on ROI coordinates from Napari.
    3. Perform nuclei segmentation using StarDist 2D versatile HE model.
    4. Extract segmentation features (centroids, labels) for each spot.
    5. Map segmented nuclei to spatial spots.
    6. Save results including segmented image, cell counts, and cell locations.
    
    Parameters
    ----------
    tissue : str
        Name of the tissue sample (used for organizing output directories)
    out_dir : str
        Base output directory path for saving results
    roi_path : str or None
        Path to ROI coordinates CSV file (from Napari). If None, processes full image.
    img_path : str
        Path to H&E stained tissue image file (supports .tif, .btf, and other formats)
    adata_path : str
        Path to ST data in AnnData format (.h5ad file)
    prob_thresh : float
        Probability threshold for StarDist segmentation (default: 0.65).
        - Lower values detect more nuclei but may include false positives.
        - Higher values are more conservative but may miss some nuclei.
        - Default values: prob_thresh=0.692478, nms_thresh=0.3
    max_cell_number : int
        Maximum number of cells/nuclei to assign per spot (default: 20).
        - If a spot contains more nuclei, only the first max_cell_number are kept.
    min_counts : int
        Minimum UMI count threshold for filtering spots (default: 500).
        - Currently not actively used but reserved for future filtering.
    
    Attributes
    ----------
    sp_adata : AnnData
        ST data with segmentation features added
    image : ImageContainer
        Squidpy ImageContainer containing the H&E image and segmentation results
    out_dir : str
        Output directory for the specific tissue sample
    loggings : Logger
        Configured logger for recording processing steps
    
    Output Files
    ------------
    - cropped_img.tif : Cropped image (if ROI mode)
    - adata_roi.h5ad : Cropped AnnData (if ROI mode)
    - nuclei_segmentation.png : Visualization of original image, segmentation, and cell counts
    - sp_adata_ns.h5ad : AnnData with segmentation features and cell locations
    - position_all_tissue_sc.csv : CSV file with cell coordinates mapped to spots
        Columns: pxl_row_in_fullres, pxl_col_in_fullres, spot_index, cell_index, cell_nums
    
    Examples
    --------
    >>> # Full image mode
    >>> ns = FineST_StarDist_nuclei(
    ...     tissue='NPC',
    ...     out_dir='./output',
    ...     roi_path=None,
    ...     img_path='./image.tif',
    ...     adata_path='./adata.h5ad',
    ...     prob_thresh=0.75,
    ...     max_cell_number=20,
    ...     min_counts=500
    ... )
    >>> ns.NucleiSegmentation()
    
    >>> # ROI mode (with region of interest)
    >>> ns = FineST_StarDist_nuclei(
    ...     tissue='CRC16_ROI',
    ...     out_dir='./output',
    ...     roi_path='./ROI_coordinates.csv',
    ...     img_path='./image.btf',
    ...     adata_path='./adata.h5ad',
    ...     prob_thresh=0.5,
    ...     max_cell_number=20,
    ...     min_counts=500
    ... )
    >>> ns.NucleiSegmentation()
    
    Notes
    -----
    - StarDist model uses CPU by default (GPU disabled via CUDA_VISIBLE_DEVICES)
    - The segmentation model '2D_versatile_he' is automatically downloaded on first use
    - Image coordinates are preserved in full resolution (pxl_row_in_fullres, pxl_col_in_fullres)
    - Cell locations are stored in both AnnData.uns['cell_locations'] and CSV format
    """

    def __init__(self, tissue, out_dir, 
                 roi_path, img_path, adata_path, 
                 prob_thresh, max_cell_number, min_counts):
        self.tissue = tissue
        self.out_dir = out_dir 

        ## my add
        self.roi_path = roi_path
        self.img_path = img_path
        self.adata_path = adata_path

        self.prob_thresh = prob_thresh
        self.max_cell_number = max_cell_number
        self.min_counts = min_counts
        os.makedirs(out_dir, exist_ok=True)
        tissue_dir = os.path.join(out_dir, tissue)
        os.makedirs(tissue_dir, exist_ok=True)

        self.out_dir = tissue_dir
        loggings = configure_logging(os.path.join(self.out_dir, 'logs'))
        self.loggings = loggings 

        if self.roi_path is not None:
            self.ST_Data, self.Img_Data = self.crop_img_adata()
            self.LoadData(self.ST_Data, self.Img_Data, self.min_counts, roi_path=self.roi_path)
        else:
            self.ST_Data = self.adata_path
            self.Img_Data = self.img_path
            self.LoadData(self.ST_Data, self.Img_Data, self.min_counts, roi_path=None)


    def create_mask(self, polygon, shape):
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
        self.loggings.info(f"polygon: \n{polygon}")
        polygon = np.clip(polygon, a_min=0, a_max=None)
        self.loggings.info(f"polygon adjusted: \n{polygon}")
        ## Keep the order of x and y unchanged
        rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], shape) 
        # rr, cc = draw.polygon(polygon.iloc[:, -2], polygon.iloc[:, -1], shape) 

        ## Set negative coordinate to 0
        mask = np.zeros(shape, dtype=bool)
        mask[rr, cc] = True
        return mask, polygon


    def crop_img_adata(self):
        """
        Crop an image and an AnnData object based on a region of interest.

        Parameters:
        roi_path : numpy.ndarray
            A numpy array specifying the region of interest.
        img_path : str
            The path to the image file.
        adata_path : str
            The path to the AnnData file.

        Returns:
        tuple
            A tuple containing the cropped image and the cropped AnnData object.
        """

        roi_coords = pd.read_csv(self.roi_path)
        self.loggings.info(f"ROI coordinates from napari package: \n{roi_coords}")

        img = io.imread(self.img_path)
        self.loggings.info(f"img shape: \n{img.shape}")

        ## Create a mask for the region of interest
        mask, roi_coords = self.create_mask(roi_coords, img.shape[:2])
        ## Find the bounding box of the region of interest
        props = measure.regionprops_table(mask.astype(int), properties=('bbox',))

        minr = props['bbox-0'][0]
        minc = props['bbox-1'][0]
        maxr = props['bbox-2'][0]
        maxc = props['bbox-3'][0]

        cropped_img = img[minr:maxr, minc:maxc]
        self.loggings.info(f"cropped_img shape: \n{cropped_img.shape}")

        ## Save the cropped image
        io.imsave(os.path.join(self.out_dir, 'cropped_img.tif'), cropped_img)

        adata = sc.read_h5ad(self.adata_path)
        # self.loggings.info(f"The adata: \n{adata}")
        spatial_range = [
            [adata.obsm["spatial"][:,0].min(), adata.obsm["spatial"][:,0].max()], 
            [adata.obsm["spatial"][:,1].min(), adata.obsm["spatial"][:,1].max()]
        ]
        self.loggings.info(f"The range of original adata: \n{spatial_range}")
        
        ## replace x and y of adata
        roi_yx = roi_coords[:, [1, 0]]   
        # roi_yx = np.stack([roi_coords.iloc[:, -1], roi_coords.iloc[:, -2]]).T
        adata_roi = adata[Path(roi_yx).contains_points(adata.obsm["spatial"]), :].copy()

        ## Update the 'spatial' field of the AnnData object
        # print(roi_coords)
        # print([roi_coords[0][1], roi_coords[0][0]])

        # Adjust spatial coordinates based on ROI
        if len(roi_coords) > 2 and roi_coords[2][0] == 0: 
            adata_roi.obsm["spatial"] = adata_roi.obsm["spatial"] - \
                                        np.array([roi_coords[0][1], 0])
        else: 
            adata_roi.obsm["spatial"] = adata_roi.obsm["spatial"] - \
                                        np.array([roi_coords[0][1], roi_coords[0][0]])

        ## Save the cropped AnnData object
        adata_roi.write(os.path.join(self.out_dir, 'adata_roi.h5ad'))
        self.loggings.info(f"The adata_roi: \n{adata_roi}")

        return adata_roi, cropped_img


    def LoadData(self, ST_Data, Img_Data, min_counts, roi_path=None):
        self.loggings.info(f'Reading spatial data: \n{ST_Data}')

        if roi_path is not None:
            sp_adata = ST_Data
        else:
            sp_adata = anndata.read_h5ad(ST_Data)
        sp_adata.obs_names_make_unique()
        sp_adata.var_names_make_unique()
        self.loggings.info(f'Spatial data shape: {sp_adata.shape}')
        # sc.pp.filter_cells(sp_adata, min_counts=min_counts)    # for FineST result
        self.loggings.info(f'Spatial data shape after QC: {sp_adata.shape}')
        self.sp_adata = sp_adata
    
        self.loggings.info(f'Reading image data: {ST_Data}')
        
        if roi_path is not None:
            img = sq.im.ImageContainer(Img_Data)
        else:
            image = plt.imread(Img_Data)
            img = sq.im.ImageContainer(image)

        crop = img.crop_corner(0, 0)
        self.loggings.info(f'Image shape: {crop.shape}')
        
        self.image = crop


    #####################################################
    # The bellow are nearly same with SpatialScope
    #####################################################
    @staticmethod
    def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
        # axis_norm = (0,1)   # normalize channels independently
        axis_norm = (0,1,2) # normalize channels jointly
        ## Make sure to normalize the input image beforehand or 
        ## supply a normalizer to the prediction function.
        ## this is the default normalizer noted in StarDist examples.
        img = normalize(img, 1, 99.8, axis=axis_norm)
        model = StarDist2D.from_pretrained('2D_versatile_he')
        labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh)
        return labels

    
    @staticmethod
    def DissectSegRes(df):
        tmps = []
        for row in df.iterrows():
            if row[1]['segmentation_label'] == 0:
                continue
            for idx,i in enumerate(row[1]['segmentation_centroid']):
                tmps.append(list(i)+[row[0],row[0]+'_{}'.format(idx),row[1]['segmentation_label']])
        return pd.DataFrame(tmps,columns=['x','y','spot_index','cell_index','cell_nums'])  

        
    def NucleiSegmentation(self):
        ## Disable GPU for StarDist (use CPU)
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        ## Pre-load the model (return value not needed, just for initialization)
        _ = StarDist2D.from_pretrained('2D_versatile_he')
        sq.im.segment(
            img=self.image,
            layer="image",
            channel=None,
            method=self.stardist_2D_versatile_he,
            layer_added='segmented_stardist_default',
            prob_thresh=self.prob_thresh)
        self.loggings.info(f"Number of segments: {len(np.unique(self.image['segmented_stardist_default']))}")
        
        ## define image layer to use for segmentation
        features_kwargs = {
            "segmentation": {
                "label_layer": "segmented_stardist_default",
                "props": ["label", "centroid"],
                "channels": [1, 2],
            }
        }
        
        ## calculate segmentation features
        sq.im.calculate_image_features(
            self.sp_adata,
            self.image,
            layer="image",
            key_added="image_features",
            features_kwargs=features_kwargs,
            features="segmentation",
            mask_circle=True,
        )
        
        df_cells = self.sp_adata.obsm['image_features'].copy().astype(object)
        for row in df_cells.iterrows():
            if row[1]['segmentation_label']>self.max_cell_number:
                df_cells.loc[row[0],'segmentation_label'] = self.max_cell_number
                ## 2024.02.27 df_cells.loc --> df_cells.at
                # df_cells.loc[row[0],'segmentation_centroid'] = row[1]['segmentation_centroid'][:self.max_cell_number]
                df_cells.at[row[0],'segmentation_centroid'] = row[1]['segmentation_centroid'][:self.max_cell_number]

        self.sp_adata.obsm['image_features'] = df_cells       
        self.sp_adata.obs["cell_count"] = self.sp_adata.obsm["image_features"]["segmentation_label"].astype(int)
        
        fig, axes = plt.subplots(1,3,figsize=(20,6),dpi=300)
        self.image.show("image", ax=axes[0])
        _ = axes[0].set_title("H&E")
        self.image.show("segmented_stardist_default", cmap="jet", interpolation="none", ax=axes[1])
        _ = axes[1].set_title("Nuclei Segmentation")
        ## 2024.11.19 add img_key=None, because the coordinates of adata.obsm['spatial'] saved the whole big figure.
        sc.pl.spatial(self.sp_adata, color=["cell_count"], img_key=None, frameon=False, ax=axes[2],title='')
        _ = axes[2].set_title("Cell Count")
        plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
        plt.close()
        
        
        self.sp_adata.uns['cell_locations'] = self.DissectSegRes(self.sp_adata.obsm['image_features'])
        self.sp_adata.obsm['image_features']['segmentation_label'] = (
            self.sp_adata.obsm['image_features']['segmentation_label'].astype(int)
        )
        self.sp_adata.obsm['image_features']['segmentation_centroid'] = (
            self.sp_adata.obsm['image_features']['segmentation_centroid'].astype(str)
        )  
        self.sp_adata.write_h5ad(os.path.join(self.out_dir, 'sp_adata_ns.h5ad'))

        #####################################
        # 2025.02.06 From '.adata' to '.csv'
        #####################################
        self.loggings.info(f"image_features: \n{self.sp_adata.obsm['image_features']}")
        coord_cell = self.sp_adata.uns['cell_locations']
        ## delete NaN: dropna()
        coord_cell_filtered = coord_cell.dropna()
        # print(coord_cell_filtered)
        ## set column names: columns()
        coord_cell_filtered.columns = ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'spot_index', 'cell_index', 'cell_nums']
        self.loggings.info(f"coord_cell_filtered: \n{coord_cell_filtered}")
        ## save coords position
        coord_cell_filtered.to_csv(os.path.join(self.out_dir, "position_all_tissue_sc.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path', default=None)

    ## my add
    parser.add_argument('--roi_path', type=str, help='ROI path', default=None)
    parser.add_argument('--adata_path', type=str, help='ST data path', default=None)
    parser.add_argument('--img_path', type=str, help='H&E stained image data path', default=None)
    
    # parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    # parser.add_argument('--Img_Data', type=str, help='H&E stained image data path', default=None)
    parser.add_argument(
        '--prob_thresh', 
        type=float, 
        help='object probability threshold, decrease this parameter if too many nucleus are missing', 
        default=0.5
    )    
    parser.add_argument('--max_cell_number', type=int, help='maximum cell number per spot', default=20)
    parser.add_argument('--min_counts', type=int, help='minimum UMI count per spot', default=500)
    args = parser.parse_args()
        
    NS = FineST_StarDist_nuclei(
        args.tissue, 
        args.out_dir, 

        ## my add
        args.roi_path,    
        args.img_path, 
        args.adata_path, 
        # args.ST_Data, 
        # args.Img_Data, 

        args.prob_thresh, 
        args.max_cell_number, 
        args.min_counts
    )

    NS.NucleiSegmentation()


##########
# NPC
##########
# conda activate FineST
# python ./demo/StarDist_nuclei_segmente.py \
#     --tissue NPC_allspot_p075 \
#     --out_dir FineST_tutorial_data/NucleiSegments \
#     --adata_path FineST_tutorial_data/SaveData/adata_imput_all_spot.h5ad \
#     --img_path FineST_tutorial_data/20210809-C-AH4199551.tif \
#     --prob_thresh 0.75

##########
# CRC16 with ROI
##########
# python ./demo/StarDist_nuclei_segmente.py \
#     --tissue CRC16um_ROI_test \
#     --out_dir ./Dataset/CRC16um/StarDist/DataOutput \
#     --roi_path ./Dataset/CRC16um/ResultsROIs/ROI4_shape.csv \
#     --adata_path ./Dataset/CRC16um/Colon_Cancer_square_016um.h5ad \
#     --img_path ./Dataset/CRC16um/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
#     --prob_thresh 0.5 \
#     --max_cell_number 20 \
#     --min_counts 500