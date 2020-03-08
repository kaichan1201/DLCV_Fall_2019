# TODO: create shell script for Problem 1
#DATA_DIR='./hw4_data/TrimmedVideos/video/valid/'
#CSV_PATH='./hw4_data/TrimmedVideos/label/gt_valid.csv'
#SAVE_DIR='./results/'
DATA_DIR=$1
CSV_PATH=$2
SAVE_DIR=$3
python3 infer_no_RNN.py --frame_num 2 --infer_data_dir ${DATA_DIR} --save_dir ${SAVE_DIR} --csv_path ${CSV_PATH}\
	                --pretrained 'best_models/no_RNN/checkpoint_best.pth.tar'
