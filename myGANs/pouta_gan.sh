#!/bin/bash

## run using command:
## $ nohup bash pouta_gan.sh > /dev/null 2>&1 &
## $ nohup bash pouta_gan.sh > check_output.out 2>&1 & # with output saved in check_output.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
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
source $HOME_DIR/miniconda3/bin/activate py39

# Run both commands simultaneously
python -u gan.py \
	--rgbDIR $datasetDIR \
	--resDIR $resultsDIR \
	--numWorkers $nW \
	--lr $learning_rate \
	--nepochs $nEpochs \
	--batchSZ $batch_size \
	--cudaNum 1 \
	--ganMethodIdx 0 >>$WDIR/trash/GANs/gan_MTHD_0.out 2>&1 &

python -u gan.py \
	--rgbDIR $datasetDIR \
	--resDIR $resultsDIR \
	--numWorkers $nW \
	--lr $learning_rate \
	--nepochs $nEpochs \
	--batchSZ $batch_size \
	--cudaNum 2 \
	--ganMethodIdx 1 >>$WDIR/trash/GANs/gan_MTHD_1.out 2>&1 &

# Wait for both background jobs to finish
wait

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"