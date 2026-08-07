## 2026.08.05 Update spot_interpolation to interpolate between-spots in x and y directions

"""
Interpolate between-spot coordinates for Visium data.

Public API::

    import FineST as fst
    fst.spot_interpolation(position_path='.../tissue_positions_list.csv')

Or from the terminal::

    python -m FineST.spot_interpolation --position_path .../tissue_positions_list.csv
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional, Tuple

import pandas as pd

from .processData import filter_pos_list, final_pos_list, inter_spot


def spot_interpolation(
    position_path: str,
    output_add: Optional[str] = None,
    output_all: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Interpolate between-spot coordinates for original Visium data.

    Loads within-spot positions, interpolates new spots in x and y directions,
    and writes CSV outputs.

    Parameters
    ----------
    position_path : str
        Path to ``tissue_positions_list.csv`` (within-spots).
    output_add : str, optional
        Output path for interpolated between-spots only.
        Default: ``tissue_positions_list_add.csv`` next to the input file.
    output_all : str, optional
        Output path for within-spots + between-spots.
        Default: ``tissue_positions_list_all.csv`` next to the input file.

    Returns
    -------
    position_add : pd.DataFrame
        Interpolated between-spots only.
    position_all : pd.DataFrame
        Original within-spots plus interpolated between-spots.
    output_add_path : str
    output_all_path : str
    """
    if not os.path.exists(position_path):
        raise FileNotFoundError(f"Position file not found: {position_path}")

    out_dir = os.path.dirname(position_path) or '.'

    try:
        position = filter_pos_list(position_path)
        if position is None or position.empty:
            raise ValueError("No valid positions found in the input file")
        print(f"Loaded {position.shape[0]} original within-spots from Visium data")
    except Exception as e:
        raise RuntimeError(f"Error loading position file: {e}") from e

    start_time = time.time()
    try:
        position_x = inter_spot(position, direction='x')
        position_y = inter_spot(position, direction='y')
        print(f"Interpolation time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        raise RuntimeError(f"Error during interpolation: {e}") from e

    try:
        position_add = final_pos_list(position_x, position_y, position=None)
        position_all = final_pos_list(position_x, position_y, position)
    except Exception as e:
        raise RuntimeError(f"Error integrating positions: {e}") from e

    output_add_path = output_add or os.path.join(out_dir, "tissue_positions_list_add.csv")
    output_all_path = output_all or os.path.join(out_dir, "tissue_positions_list_all.csv")

    try:
        position_add.to_csv(output_add_path, index=True)
        position_all.to_csv(output_all_path, index=True)
        print(f"Saved interpolated spots to: {output_add_path}")
        print(f"Saved all spots to: {output_all_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving output files: {e}") from e

    ratio_add = round(position_add.shape[0] / position.shape[0], 3)
    ratio_all = round(position_all.shape[0] / position.shape[0], 3)
    print(f"# of interpolated between-spots: {ratio_add} times vs. original within-spots")
    print(f"# of final all spots: {ratio_all} times vs. original within-spots")

    return position_add, position_all, output_add_path, output_all_path


main = spot_interpolation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate between spots in x and y directions for Visium data'
    )
    parser.add_argument(
        '--position_path',
        required=True,
        help='Full path to the position list file (e.g., .../tissue_positions_list.csv)',
    )
    parser.add_argument(
        '--output_add',
        default=None,
        help='Optional output path for interpolated spots only',
    )
    parser.add_argument(
        '--output_all',
        default=None,
        help='Optional output path for all spots',
    )
    args = parser.parse_args()
    spot_interpolation(args.position_path, args.output_add, args.output_all)


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