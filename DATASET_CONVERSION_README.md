# 数据集转换说明

## 概述
这个脚本用于将 `allenai/tulu-3-sft-mixture` 数据集转换为Qwen微调所需的格式。

## 文件说明
- `convert_tulu_dataset.py`: 主要转换脚本
- `run_conversion.sh`: 简化的运行脚本
- `tulu_conversations.json`: 转换后的数据文件（运行后生成）

## 使用方法

### 方法1: 使用简化脚本
```bash
bash run_conversion.sh
```

### 方法2: 直接运行Python脚本
```bash
# 安装依赖
pip install datasets tqdm

# 运行转换
python convert_tulu_dataset.py --dataset allenai/tulu-3-sft-mixture --output tulu_conversations.json
```

### 方法3: 自定义参数
```bash
python convert_tulu_dataset.py \
    --dataset allenai/tulu-3-sft-mixture \
    --output my_conversations.json \
    --max_samples 1000
```

## 参数说明
- `--dataset`: 数据集名称（默认: allenai/tulu-3-sft-mixture）
- `--output`: 输出文件路径（默认: tulu_conversations.json）
- `--max_samples`: 最大样本数量（可选，用于测试）

## 输出格式
转换后的数据格式符合Qwen微调要求：
```json
[
  {
    "id": "tulu_0",
    "conversations": [
      {
        "from": "user",
        "value": "用户问题"
      },
      {
        "from": "assistant", 
        "value": "助手回答"
      }
    ]
  }
]
```

## 在微调中使用
转换完成后，在微调脚本中指定数据路径：

```bash
# 使用命令行参数
bash finetune/finetune_ds.sh -d $(pwd)/tulu_conversations.json

# 或者修改finetune_ds.sh中的DATA变量
DATA="$(pwd)/tulu_conversations.json"
```

## 注意事项
1. 确保有足够的磁盘空间存储转换后的数据
2. 转换过程可能需要一些时间，取决于数据集大小
3. 脚本会自动处理不同的消息格式和角色映射
4. 只保留包含有效对话的样本（至少包含user和assistant的对话）

## 故障排除
如果遇到问题：
1. 检查网络连接（需要下载数据集）
2. 确保安装了必要的依赖包
3. 检查输出目录的写入权限
4. 查看控制台输出的错误信息


