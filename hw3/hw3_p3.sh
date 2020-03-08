#!/bin/bash
TGT_IMG_DIR=$1
TGT_NAME=$2
OUT_CSV_PATH=$3
if [ "${TGT_NAME}" == "mnistm" ]; then
	python3 infer_DANN.py --pretrained 'best_models/DANN/s_m/DANN_base/model_best.pth.tar' \
		              --tgt 'mnistm' --infer_data_dir $TGT_IMG_DIR \
	        	      --csv_path $OUT_CSV_PATH
fi

if [ "${TGT_NAME}" == "svhn" ]; then
	python3 infer_DANN.py --pretrained 'best_models/DANN/m_s/DANN_base/model_best.pth.tar' \
        	              --tgt 'svhn' --infer_data_dir $TGT_IMG_DIR \
                	      --csv_path $OUT_CSV_PATH
fi
