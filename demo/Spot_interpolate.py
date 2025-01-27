## 2024.10.30 related to /docs/source/Between_spot_demo.ipynb

import os
import time
import argparse
import FineST as fst


def main(data_path, dataset, position_list):

    # Add these lines before you save the csv
    dir_path = os.path.join(data_path, dataset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    os.chdir(dir_path)
    
    # 1. Load and filter original position list 
    position = fst.filter_pos_list(position_list)

    # 2. Interpolate 'between spots' in horizontal and vertical directions 
    start_time = time.time()

    position_x = fst.inter_spot(position, direction='x')
    position_y = fst.inter_spot(position, direction='y')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The spots feature interpolation time is: {execution_time} seconds")

    # 3. Integrate 'between spot' and 'within spot'
    position_add = fst.final_pos_list(position_x, position_y, position=None)
    position_all = fst.final_pos_list(position_x, position_y, position)

        
    # save position list to .csv file
    position_add.to_csv(f"{dataset}_position_add_tissue.csv") 
    position_all.to_csv(f"{dataset}_position_all_tissue.csv")    

    # Calculate the ratios
    ratio_add = round(position_add.shape[0] / position.shape[0], 3)
    ratio_all = round(position_all.shape[0] / position.shape[0], 3)

    # Print the results of spot numbers
    print(f"# of interpolated between-spots are: {ratio_add} times vs. original within-spots")
    print(f"# 0f final all spots are: {ratio_all} times vs. original within-spots")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Data path for input and output')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--position_list', required=True, 
                        help='Position file name, ‘tissue_positions_list.csv’ in spatial folder')
    args = parser.parse_args()

    main(args.data_path, args.dataset, args.position_list)


# python ./FineST/demo/Spot_interpolate.py \
#     --data_path ./Dataset/NPC/ \
#     --position_list tissue_positions_list.csv \
#     --dataset patient1 