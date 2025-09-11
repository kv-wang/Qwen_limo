#!/bin/bash

# 转换Tulu数据集为Qwen微调格式的脚本

echo "开始转换allenai/tulu-3-sft-mixture数据集..."

# 安装必要的依赖
echo "安装依赖包..."
pip install datasets tqdm

# 运行转换脚本
echo "运行数据转换..."
python convert_tulu_dataset.py --dataset allenai/tulu-3-sft-mixture --output tulu_conversations.json

echo "转换完成！"
echo "生成的文件: tulu_conversations.json"
echo ""
echo "使用方法:"
echo "1. 在微调脚本中指定数据路径:"
echo "   bash finetune/finetune_ds.sh -d $(pwd)/tulu_conversations.json"
echo ""
echo "2. 或者修改finetune_ds.sh中的DATA变量:"
echo "   DATA=\"$(pwd)/tulu_conversations.json\""


