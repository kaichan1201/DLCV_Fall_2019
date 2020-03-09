IMGNAME=0224.png
MODE=val
IMG=../hw2_data/$MODE/img/$IMGNAME
GT=../hw2_data/$MODE/seg/$IMGNAME
#python3 infer.py --infer_img $IMG --pretrained model_aspp_small_lr/model_best.pth.tar --use_better
python3 infer.py --infer_img $IMG --pretrained model_base_big_decrease_lr/model_best.pth.tar
python3 viz_mask.py --img_path $IMG --seg_path infer.png --out_path exp.png
python3 viz_mask.py --img_path $IMG --seg_path $GT       --out_path gt.png
