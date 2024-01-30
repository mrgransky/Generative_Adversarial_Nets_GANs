#!/bin/bash

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"

datasetDIR="$HOME_DIR/datasets/sentinel2-l1c_RGB_IMGs"
resultsDIR="$HOME_DIR/trash_logs/GANs/misc" ########## must be adjusted! ##########

# Activate Conda environment
source activate py39

for gan_idx in 0 1
do
	echo "GAN Method IDX: $gan_idx"
	python -u gan.py \
		--rgbDIR $datasetDIR \
		--resDIR $resultsDIR \
		--numWorkers 20 \
		--lr 0.0003 \
		--nepochs 50 \
		--batchSZ 8 \
		--ganMethodIdx $gan_idx > $resultsDIR/gan_method_$gan_idx.out 2>&1
done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"

# Deactivate Conda environment
conda deactivate