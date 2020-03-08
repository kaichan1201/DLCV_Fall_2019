#!/bin/bash
TGT_IMG_DIR=$1
TGT_NAME=$2
OUT_CSV_PATH=$3
if [ "$TGT_NAME" == "mnistm" ]; then
	python3 infer_ADDA.py --pretrained 'best_models/ADDA/s_m/base_dropout_noisy_0.1/checkpoint_best.pth.tar' \
		              --pretrained_src 'best_models/ADDA/s_m/src_only_big_E_dr/checkpoint_best.pth.tar' \
	        	      --tgt 'mnistm' --infer_data_dir $TGT_IMG_DIR \
	              	      --csv_path $OUT_CSV_PATH
fi

if [ "$TGT_NAME" == "svhn" ]; then
	python3 infer_ADDA.py --pretrained 'best_models/ADDA/m_s/base_dropout_test/checkpoint_best.pth.tar' \
        	              --pretrained_src 'best_models/ADDA/m_s/src_only_big_E_dr/checkpoint_best.pth.tar' \
                	      --tgt 'svhn' --infer_data_dir $TGT_IMG_DIR \
                              --csv_path $OUT_CSV_PATH
fi
