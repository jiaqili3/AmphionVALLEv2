#!/usr/bin/env bash
#SBATCH --job-name=train-valle-ar            # Job name
#SBATCH --nodes=1                  # Run all processes on a single node  
#SBATCH --ntasks=1                 # Run a single task        
#SBATCH --cpus-per-task=32         # Number of CPU cores per task
#SBATCH --gres=gpu:4               # Number of GPU cores per node
#SBATCH --partition=mm_steel
export PYTHONPATH="./"
 
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Number of CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo ""
echo "Running script... "
 
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
 

######## Build Experiment Environment ###########
exp_dir="/mnt/workspace/lizhekai/AmphionVALLEv2/egs/tts/valle_gpt_simple"
echo exp_dir
work_dir="/mnt/workspace/lizhekai/AmphionVALLEv2/"
echo work_dir


export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

######## Set Config File Dir ##############
if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_nar_libritts.json
fi
echo "Exprimental Configuration File: $exp_config"

######## Set the experiment name ##########
exp_name="nar_libritts_dev_clean"

port=17004

######## Train Model ###########
echo "Experimental Name: $exp_name"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port $port "${work_dir}"/bins/tts/train.py --config $exp_config --exp_name $exp_name --log_level debug 
    # --resume \
    # --resume_type "resume"

# uncomment the "resume" part to automatically resume from the last-time checkpoint