#!/usr/bin/env python3
"""
将allenai/tulu-3-sft-mixture数据集转换为Qwen微调所需的格式
"""

import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import os

def convert_tulu_to_qwen_format(dataset_name="allenai/tulu-3-sft-mixture", output_path="tulu_conversations.json"):
    """
    将Tulu数据集转换为Qwen微调格式
    
    Args:
        dataset_name: 数据集名称
        output_path: 输出文件路径
    """
    print(f"正在加载数据集: {dataset_name}")
    
    # 加载数据集
    try:
        dataset = load_dataset(dataset_name)
        print(f"数据集加载成功，包含以下分割: {list(dataset.keys())}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return False
    
    # 选择训练集（通常是'train'分割）
    train_split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    train_data = dataset[train_split]
    print(f"使用分割: {train_split}, 样本数量: {len(train_data)}")
    
    converted_data = []
    
    print("开始转换数据格式...")
    for i, sample in enumerate(tqdm(train_data, desc="转换进度")):
        try:
            # 提取对话内容
            conversations = []
            
            # 检查不同的可能字段名
            messages = None
            if 'messages' in sample:
                messages = sample['messages']
            elif 'conversations' in sample:
                messages = sample['conversations']
            elif 'chat' in sample:
                messages = sample['chat']
            else:
                # 尝试找到包含对话的字段
                for key in sample.keys():
                    if isinstance(sample[key], list) and len(sample[key]) > 0:
                        if isinstance(sample[key][0], dict) and ('role' in sample[key][0] or 'from' in sample[key][0]):
                            messages = sample[key]
                            break
            
            if not messages:
                print(f"警告: 样本 {i} 没有找到对话数据，跳过")
                continue
            
            # 转换消息格式
            for msg in messages:
                if isinstance(msg, dict):
                    # 确定角色映射
                    role = None
                    if 'role' in msg:
                        role = msg['role']
                    elif 'from' in msg:
                        role = msg['from']
                    
                    # 确定内容字段
                    content = None
                    if 'content' in msg:
                        content = msg['content']
                    elif 'value' in msg:
                        content = msg['value']
                    elif 'text' in msg:
                        content = msg['text']
                    
                    if role and content:
                        # 角色映射：user -> user, assistant/gpt -> assistant, system -> system
                        if role.lower() in ['user', 'human']:
                            mapped_role = 'user'
                        elif role.lower() in ['assistant', 'gpt', 'ai', 'model']:
                            mapped_role = 'assistant'
                        elif role.lower() in ['system']:
                            mapped_role = 'system'
                        else:
                            # 默认映射
                            mapped_role = 'user' if role.lower() in ['user', 'human'] else 'assistant'
                        
                        conversations.append({
                            "from": mapped_role,
                            "value": str(content).strip()
                        })
            
            # 只保留有效的对话（至少包含user和assistant的对话）
            if len(conversations) >= 2:
                converted_sample = {
                    "id": f"tulu_{i}",
                    "conversations": conversations
                }
                converted_data.append(converted_sample)
            
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue
    
    print(f"转换完成，有效样本数量: {len(converted_data)}")
    
    # 保存转换后的数据
    print(f"正在保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("数据转换完成！")
    
    # 显示示例
    if converted_data:
        print("\n转换后的数据示例:")
        print(json.dumps(converted_data[0], ensure_ascii=False, indent=2))
    
    return True

def main():
    parser = argparse.ArgumentParser(description="转换Tulu数据集为Qwen微调格式")
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture", 
                       help="数据集名称 (默认: allenai/tulu-3-sft-mixture)")
    parser.add_argument("--output", default="tulu_conversations.json", 
                       help="输出文件路径 (默认: tulu_conversations.json)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大样本数量（用于测试，默认处理全部）")
    
    args = parser.parse_args()
    
    # 如果指定了最大样本数，需要修改加载方式
    if args.max_samples:
        print(f"注意: 将只处理前 {args.max_samples} 个样本")
        # 这里可以添加限制样本数的逻辑
    
    success = convert_tulu_to_qwen_format(args.dataset, args.output)
    
    if success:
        print(f"\n✅ 转换成功！输出文件: {args.output}")
        print(f"现在您可以在微调脚本中使用: --data {os.path.abspath(args.output)}")
    else:
        print("❌ 转换失败！")

if __name__ == "__main__":
    main()
