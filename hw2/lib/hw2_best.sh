RESUME='best_models/model_aspp_dec_res_best.pth.tar'

IMG_DIR=$1
SEG_DIR=$2
#IMG_DIR='../hw2_data/val/img/'
#SEG_DIR='../hw2_data/val/my_seg/'
#GT_DIR='../hw2_data/val/seg/'

python3 infer_dir.py --pretrained $RESUME  --data_dir $IMG_DIR --save_dir $SEG_DIR --use_model 'ASPP_DEC_RES'

#python3 mean_iou_evaluate.py -g $GT_DIR -p $SEG_DIR
