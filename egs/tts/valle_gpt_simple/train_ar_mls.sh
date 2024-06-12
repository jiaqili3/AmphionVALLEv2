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
exp_dir="/mnt/petrelfs/hehaorui/jiaqi/AmphionVALLEv2/egs/tts/valle_gpt_simple"
echo exp_dir
work_dir="/mnt/petrelfs/hehaorui/jiaqi/AmphionVALLEv2"
echo work_dir


export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_ar_mls.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="ar_mls_speechtokenizer"

port=53333


######## Train Model ###########
echo "Exprimental Name: $exp_name"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $port \
"${work_dir}"/bins/tts/train.py \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    --seed $RANDOM \
    --resume \
    --resume_type "resume" \
    $1

# uncomment the "resume" part to automatically resume from the last-time checkpoint

sleep 60

echo "Resuming......"

while [ ! -f "/mnt/petrelfs/hehaorui/jiaqi/stop_training" ]; do
    port=$((port + 1))
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $port \
    "${work_dir}"/bins/tts/train.py \
        --config $exp_config \
        --exp_name $exp_name \
        --log_level debug \
        --seed $RANDOM \
        --resume \
        --resume_type "resume"
    
    sleep 60
done