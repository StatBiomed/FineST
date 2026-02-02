import os
import time
import argparse
import FineST as fst

def main(position_path):
    # Get directory of the input file
    out_dir = os.path.dirname(position_path)

    # 1. Load and filter the original position list
    position = fst.filter_pos_list(position_path)

    # 2. Interpolate "between spots" in x and y directions
    start_time = time.time()
    position_x = fst.inter_spot(position, direction='x')
    position_y = fst.inter_spot(position, direction='y')
    print(f"Interpolation time: {time.time() - start_time:.2f} seconds")

    # 3. Integrate interpolated and original spots
    position_add = fst.final_pos_list(position_x, position_y, position=None)
    position_all = fst.final_pos_list(position_x, position_y, position)

    # 4. Save results to CSV files in the same directory as the input file
    position_add.to_csv(os.path.join(out_dir, "tissue_positions_list_add.csv"))
    position_all.to_csv(os.path.join(out_dir, "tissue_positions_list_all.csv"))

    # 5. Print ratios of interpolated and total spots to the original number
    ratio_add = round(position_add.shape[0] / position.shape[0], 3)
    ratio_all = round(position_all.shape[0] / position.shape[0], 3)
    print(f"# of interpolated between-spots: {ratio_add} times vs. original within-spots")
    print(f"# of final all spots: {ratio_all} times vs. original within-spots")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--position_path', required=True, help='Full path to the position list file (e.g., .../tissue_positions_list.csv)')
    args = parser.parse_args()

    main(args.position_path)



#######################
# usage
#######################
# cd ~/FineST_demo
# conda activate FineST
# python ./demo/Spot_interpolation.py --position_path FineST_tutorial_data/spatial/tissue_positions_list.csv