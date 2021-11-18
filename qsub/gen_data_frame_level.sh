#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N fyj_gen_data_frame_level_train_301

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=gpu03

# remember to activate your conda env
source activate nnsslm_env

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py

# python ../gen_multi_sources_frame_level_data.py "/CDShare2/SSLR/lsp_test_library_w8192/gt_frame" \
#                                                 "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_library/audio" \
#                                                 "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level_stft"

# python ../gen_multi_sources_frame_level_data.py "/CDShare2/SSLR/lsp_test_106_w8192/gt_frame" \
#                                                 "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_106/audio" \
#                                                 "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level_stft"

# python ../gen_multi_sources_frame_level_data.py "/CDShare2/SSLR/lsp_train_106_w8192/gt_frame" \
#                                                 "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/lsp_train_106/audio" \
#                                                 "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/train_data_frame_level_stft"                                                

python ../gen_multi_sources_frame_level_data.py "/CDShare2/SSLR/lsp_train_301_w8192/gt_frame" \
                                                "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/lsp_train_301/audio" \
                                                "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/train_data_frame_level_stft"  

#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
