# TODO: create shell script for Problem 3
#DATA_DIR='./hw4_data/FullLengthVideos/videos/valid/'
#SAVE_DIR='./results'
DATA_DIR=$1
SAVE_DIR=$2
python3 infer_RNN_seq.py --infer_data_dir ${DATA_DIR} --save_dir ${SAVE_DIR} \
			 --pretrained './best_models/RNN_seq/checkpoint_best.pth.tar'
