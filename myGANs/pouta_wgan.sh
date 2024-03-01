#!/bin/bash

## run using command:
## $ nohup bash pouta_wgan.sh > /dev/null 2>&1 &
## $ nohup bash pouta_wgan.sh > check_output_wgan.out 2>&1 & # with output saved in check_output.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
source $HOME_DIR/miniconda3/bin/activate py39

WDIR="/media/volume"
echo "HOME DIR $HOME_DIR | WDIR: $WDIR"
# datasetDIR="$HOME_DIR/datasets/sentinel2-l1c_RGB_IMGs"
# resultsDIR="$HOME_DIR/trash_logs/GANs/misc" ########## must be adjusted! ##########
datasetDIR="$WDIR/datasets/sentinel2-l1c_RGB_IMGs"
resultsDIR="$WDIR/trash/GANs/misc" ########## must be adjusted! ##########
nW=24
batch_size=4
nEpochs=50
learning_rate=0.0003
zero_centered_GP=true
spectral_norm_critic=true
spectral_norm_generator=true

# python my_file.py --zeroCenteredGP $zero_centered_GP --spectralNormCritic $spectral_norm_critic

python -u wgan.py \
	--rgbDIR $datasetDIR \
	--resDIR $resultsDIR \
	--numWorkers $nW \
	--zeroCenteredGP $zero_centered_GP \
	--spectralNormCritic $spectral_norm_critic \
	--spectralNormGen $spectral_norm_generator \
	--lr $learning_rate \
	--nepochs $nEpochs \
	--batchSZ $batch_size \
	--cudaNum 2 \
	--wganMethodIdx 1 >>$WDIR/trash/GANs/wgan-gp_optim_vals.out 2>&1 &

done_txt="$user finished job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"