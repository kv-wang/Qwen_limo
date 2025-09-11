#!/bin/bash

# Qwen2.5-7B Full Fine-tuning with FSDP on 8 GPUs
# 使用8卡进行Qwen2.5-7B模型的FSDP全参数微调

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 模型和数据路径配置
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B"  # 或者使用本地路径
DATA_PATH="./tulu_conversations.json"     # 训练数据路径
OUTPUT_DIR="./output_qwen2.5_7b_fsdp" # 输出目录
# EVAL_DATA_PATH="./data/eval_data.json" # 验证数据路径（可选） 暂时不启动验证

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练参数配置
BATCH_SIZE=1                    # 每设备批次大小
GRADIENT_ACCUMULATION_STEPS=8   # 梯度累积步数
LEARNING_RATE=1e-5              # 学习率
NUM_EPOCHS=3                    # 训练轮数
MAX_LENGTH=2048                 # 最大序列长度
WARMUP_RATIO=0.03               # 预热比例
WEIGHT_DECAY=0.1                # 权重衰减
ADAM_BETA2=0.95                 # Adam优化器beta2参数

# FSDP配置
FSDP_CONFIG="full_shard auto_wrap"  # FSDP配置：全分片和自动包装

# 日志和保存配置
SAVE_STEPS=500                  # 保存步数间隔
LOGGING_STEPS=10                # 日志记录步数间隔
EVAL_STEPS=500                  # 评估步数间隔
SAVE_TOTAL_LIMIT=3              # 最多保存的检查点数量

# 启动训练
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    finetune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta2 $ADAM_BETA2 \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --bf16 True \
    --dataloader_num_workers 4 \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --evaluation_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --report_to "tensorboard" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --fsdp "$FSDP_CONFIG" \
    --fsdp_config "fsdp_config.json" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --optim "adamw_torch" \
    --seed 42

echo "训练完成！模型保存在: $OUTPUT_DIR"
