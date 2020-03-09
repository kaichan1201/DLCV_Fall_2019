RESUME='model_aspp_dec_res/model_best.pth.tar'
python3 test.py --pretrained $RESUME  --data_dir '../hw2_data' --use_model 'ASPP_DEC_RES'

RESUME='model_base_big_decrease_lr/model_best.pth.tar'
python3 test.py --pretrained $RESUME  --data_dir '../hw2_data' --use_model 'BASE'
