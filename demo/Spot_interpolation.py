import os
import time
import argparse
from typing import Optional
import FineST as fst


def main(position_path: str, output_add: Optional[str] = None, 
         output_all: Optional[str] = None) -> None:
    """
    Interpolate between spots for original Visium data.
    
    This function performs interpolation on the original Visium spot positions to create
    additional "between-spots" in both horizontal (x-direction) and vertical (y-direction)
    directions. This increases the spatial resolution by adding interpolated spots between
    the original "within-spots" measured by Visium.
    
    The interpolation process:
    1. Loads the original Visium spot positions (within-spots)
    2. Interpolates new spots in the horizontal (x) direction between existing spots
    3. Interpolates new spots in the vertical (y) direction between existing spots
    4. Combines all interpolated spots and optionally merges with original spots

    
    Parameters
    ----------
    position_path : str
        Full path to the Visium position list file (typically tissue_positions_list.csv)
        This file contains the original "within-spots" positions measured by Visium.
    output_add : str, optional
        Optional output path for interpolated spots only (between-spots).
        Defaults to 'tissue_positions_list_add.csv' in the same directory as the input file.
    output_all : str, optional
        Optional output path for all spots (original within-spots + interpolated between-spots).
        Defaults to 'tissue_positions_list_all.csv' in the same directory as the input file.
    
    Returns
    -------
    None
        Outputs are saved to CSV files. Prints statistics about the interpolation results.
    
    Output Files
    ------------
    - tissue_positions_list_add.csv: Contains only the interpolated between-spots
    - tissue_positions_list_all.csv: Contains both original within-spots and interpolated between-spots
    
    Notes
    -----
    - This function is specifically designed for Visium data format
    - The interpolation increases the number of spots by approximately 2-3x
    - Interpolated spots are positioned between original spots in both x and y directions
    - The output can be used as input for image feature extraction at sub-spot or single-nuclei resolution
    
    Examples
    --------
    >>> # Basic usage
    >>> main('tissue_positions_list.csv')
    
    >>> # With custom output paths
    >>> main(
    ...     'tissue_positions_list.csv',
    ...     output_add='between_spots.csv',
    ...     output_all='all_spots.csv'
    ... )
    """
    # Validate input file exists
    if not os.path.exists(position_path):
        raise FileNotFoundError(f"Position file not found: {position_path}")
    
    # Get directory of the input file
    out_dir = os.path.dirname(position_path)
    if not out_dir:
        out_dir = '.'  # Current directory if no path specified
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load and filter the original Visium spot positions (within-spots)
    #    These are the original spots measured by Visium technology
    try:
        position = fst.filter_pos_list(position_path)
        if position is None or position.empty:
            raise ValueError("No valid positions found in the input file")
        print(f"Loaded {position.shape[0]} original within-spots from Visium data")
    except Exception as e:
        raise RuntimeError(f"Error loading position file: {e}") from e

    # 2. Interpolate "between-spots" in horizontal (x) and vertical (y) directions
    #    This creates new spots between the original Visium spots to increase spatial resolution
    #    - Horizontal interpolation: adds spots between spots in the x-direction
    #    - Vertical interpolation: adds spots between spots in the y-direction
    start_time = time.time()
    try:
        position_x = fst.inter_spot(position, direction='x')  # Horizontal interpolation
        position_y = fst.inter_spot(position, direction='y')  # Vertical interpolation
        interpolation_time = time.time() - start_time
        print(f"Interpolation time: {interpolation_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error during interpolation: {e}") from e

    # 3. Integrate interpolated spots
    #    - position_add: Contains only the interpolated between-spots (from x and y directions)
    #    - position_all: Contains both original within-spots and interpolated between-spots
    try:
        position_add = fst.final_pos_list(position_x, position_y, position=None)
        position_all = fst.final_pos_list(position_x, position_y, position)
    except Exception as e:
        raise RuntimeError(f"Error integrating positions: {e}") from e

    # 4. Save results to CSV files
    # Note: Save with index=True to include index column (required for Image_feature_extraction.py)
    output_add_path = output_add or os.path.join(out_dir, "tissue_positions_list_add.csv")
    output_all_path = output_all or os.path.join(out_dir, "tissue_positions_list_all.csv")
    
    try:
        position_add.to_csv(output_add_path, index=True)
        position_all.to_csv(output_all_path, index=True)
        print(f"Saved interpolated spots to: {output_add_path}")
        print(f"Saved all spots to: {output_all_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving output files: {e}") from e

    # 5. Print ratios of interpolated and total spots to the original number
    ratio_add = round(position_add.shape[0] / position.shape[0], 3)
    ratio_all = round(position_all.shape[0] / position.shape[0], 3)
    print(f"# of interpolated between-spots: {ratio_add} times vs. original within-spots")
    print(f"# of final all spots: {ratio_all} times vs. original within-spots")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate between spots in x and y directions for Visium data'
    )
    parser.add_argument(
        '--position_path', 
        required=True, 
        help='Full path to the position list file (e.g., .../tissue_positions_list.csv)'
    )
    parser.add_argument(
        '--output_add',
        default=None,
        help='Optional output path for interpolated spots only (default: same directory as input)'
    )
    parser.add_argument(
        '--output_all',
        default=None,
        help='Optional output path for all spots (default: same directory as input)'
    )
    args = parser.parse_args()

    main(args.position_path, args.output_add, args.output_all)



#######################
# Usage examples
#######################
## Basic usage (output files will be saved in the same directory as input):
# cd ~/FineST_demo
# conda activate FineST
# python ./demo/Spot_interpolation.py \
#    --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv

## With custom output paths:
# python ./demo/Spot_interpolation.py \
#    --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv \
#    --output_add FineST_tutorial_data/spatial/tissue_positions_list_add.csv \
#    --output_all FineST_tutorial_data/spatial/tissue_positions_list_all.csv