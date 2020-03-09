#RESUME='model_aspp_dec_res/model_best.pth.tar'
RESUME='model_base_big_decrease_lr/model_best.pth.tar'

IMG_DIR='test_dir/img'
SEG_DIR='test_dir/seg'
VIS_DIR='test_dir/vis_base'
mkdir ${VIS_DIR}
python3 infer_dir.py --pretrained $RESUME  --data_dir $IMG_DIR --save_dir $SEG_DIR --use_model 'BASE'

for FILE in $IMG_DIR/*; do
	echo "visualizing ${FILE##*/}"
	python3 viz_mask.py --img_path ${IMG_DIR}/${FILE##*/} --seg_path ${SEG_DIR}/${FILE##*/}\
		--out_path ${VIS_DIR}/${FILE##*/}
done
