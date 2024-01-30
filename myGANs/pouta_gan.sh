#!/bin/bash

## run using command:
## $ nohup bash pouta_gan.sh > /dev/null 2>&1 &
## $ nohup bash pouta_gan.sh > check_output.out 2>&1 &

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
source $HOME_DIR/miniconda3/bin/activate py39

# for gan_idx in 0 1
# do
# 	echo "GAN Method IDX: $gan_idx"
# 	python -u gan.py \
# 		--rgbDIR $datasetDIR \
# 		--resDIR $resultsDIR \
# 		--numWorkers 20 \
# 		--lr 0.0003 \
# 		--nepochs 50 \
# 		--batchSZ 8 \
# 		--ganMethodIdx $gan_idx >>$HOME_DIR/trash_logs/GANs/gan_method_$gan_idx.out 2>&1
# done

# # # Run both commands simultaneously
# python -u gan.py \
# 	--rgbDIR $datasetDIR \
# 	--resDIR $resultsDIR \
# 	--numWorkers 20 \
# 	--lr 0.0003 \
# 	--nepochs 50 \
# 	--batchSZ 8 \
# 	--ganMethodIdx 0 >>$HOME_DIR/trash_logs/GANs/gan_method_0.out 2>&1 &

# python -u gan.py \
# 	--rgbDIR $datasetDIR \
# 	--resDIR $resultsDIR \
# 	--numWorkers 20 \
# 	--lr 0.0003 \
# 	--nepochs 50 \
# 	--batchSZ 8 \
# 	--ganMethodIdx 1 >>$HOME_DIR/trash_logs/GANs/gan_method_1.out 2>&1 &

# # Wait for both background jobs to finish
# wait

# Run the commands in parallel
echo "0 1" | parallel -j 2 python -u gan.py \
	--rgbDIR $datasetDIR \
	--resDIR $resultsDIR \
	--numWorkers 20 \
	--lr 0.0003 \
	--nepochs 50 \
	--batchSZ 8 \
	--ganMethodIdx {} >> $HOME_DIR/trash_logs/GANs/gan_method_{}.out 2>&1

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"