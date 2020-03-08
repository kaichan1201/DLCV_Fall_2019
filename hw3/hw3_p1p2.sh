#!/bin/bash
SAVE_DIR=$1
python3 infer_GAN.py --pretrained_G 'best_models/DCGAN/GAN_fc_noisy/G_100.pth.tar' \
	             --infer_GAN_mode 'DC' --save_dir $SAVE_DIR --random_seed 1000
python3 infer_GAN.py --pretrained_G 'best_models/ACGAN/AC_fc_noisy_soft/G_100.pth.tar' \
	             --infer_GAN_mode 'AC' --save_dir $SAVE_DIR --random_seed 100
