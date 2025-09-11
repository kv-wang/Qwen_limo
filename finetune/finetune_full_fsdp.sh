#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers using FSDP (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="tulu_conversations.json"

# FSDP Configuration
FSDP_CONFIG="full_shard"  # FSDP configuration: full sharding
FSDP_CONFIG_FILE="fsdp_config.json"  # FSDP configuration file

function usage() {
    echo '
Usage: bash finetune/finetune_full_fsdp.sh [-m MODEL_PATH] [-d DATA_PATH] [-o OUTPUT_DIR] [-opt OPTIMIZER]
    -m, --model: Model path (default: Qwen/Qwen2.5-7B)
    -d, --data: Training data path
    -o, --output: Output directory (default: output_qwen_fsdp)
    -opt, --optimizer: Optimizer type (adamw_torch, muon, adamuon, limo)
    -h, --help: Show this help message
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -o | --output )
            shift
            OUTPUT_DIR=$1
            ;;
        -opt | --optimizer )
            shift
            OPTIMIZER=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# Set default optimizer if not provided
OPTIMIZER=${OPTIMIZER:-"muon"} # adamw_torch, muon, adamuon, limo

# Set default output directory if not provided (include optimizer name)
OUTPUT_DIR=${OUTPUT_DIR:-"output_qwen_fsdp_${OPTIMIZER}"}

# Create output directory
mkdir -p $OUTPUT_DIR

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Check if FSDP config file exists, if not create a default one
if [ ! -f "$FSDP_CONFIG_FILE" ]; then
    echo "Creating default FSDP configuration file: $FSDP_CONFIG_FILE"
    cat > $FSDP_CONFIG_FILE << EOF
{
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "fsdp_backward_prefetch": "BACKWARD_PRE",
    "fsdp_cpu_ram_efficient_loading": "true",
    "fsdp_forward_prefetch": "false",
    "fsdp_offload_params": "false",
    "fsdp_sharding_strategy": "FULL_SHARD",
    "fsdp_state_dict_type": "FULL_STATE_DICT",
    "fsdp_sync_module_states": "true",
    "fsdp_use_orig_params": "true"
}
EOF
fi

# Training arguments for FSDP full fine-tuning
TRAINING_ARGS="
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --bf16 True \
    --dataloader_num_workers 2 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 10 \
    --evaluation_strategy no \
    --logging_steps 10 \
    --report_to tensorboard \
    --logging_dir $OUTPUT_DIR/logs \
    --fsdp $FSDP_CONFIG \
    --fsdp_config $FSDP_CONFIG_FILE \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --optim $OPTIMIZER \
    --seed 42 \
    --max_grad_norm 1.0 \
    --dataloader_pin_memory False \
    --dataloader_persistent_workers False
"

# Add optimizer-specific parameters
if [ "$OPTIMIZER" = "muon" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --use_muon True"
elif [ "$OPTIMIZER" = "adamuon" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --use_adamuon True"
elif [ "$OPTIMIZER" = "limo" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --use_limo True"
fi


echo "Starting FSDP full fine-tuning with the following configuration:"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Output: $OUTPUT_DIR"
echo "Optimizer: $OPTIMIZER"
echo "FSDP Config: $FSDP_CONFIG"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Number of nodes: $NNODES"
echo ""

torchrun $DISTRIBUTED_ARGS finetune.py $TRAINING_ARGS

echo "Training completed! Model saved in: $OUTPUT_DIR"
