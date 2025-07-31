#!/bin/bash

echo "🚀 Qwen-3 8B 分布式训练启动脚本"
echo "=================================="

# 检查GPU数量
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到。需要CUDA支持。"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "🔍 检测到 $NUM_GPUS 个GPU"

# 显示GPU信息
echo "📊 GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# 检查必要文件
if [ ! -f "training_data.json" ]; then
    echo "❌ training_data.json 未找到。请先运行 data_preprocessing.py"
    exit 1
fi

echo "✅ training_data.json 找到"

# 默认参数
BATCH_SIZE=${BATCH_SIZE:-2}           # 每个GPU的批次大小
GRAD_ACCUM=${GRAD_ACCUM:-4}           # 梯度累积步数
EPOCHS=${EPOCHS:-3}                   # 训练轮数
LEARNING_RATE=${LEARNING_RATE:-5e-5}  # 学习率
OUTPUT_DIR=${OUTPUT_DIR:-"./qwen_advertising_copy_distributed"}

# 计算有效批次大小
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo ""
echo "🔧 训练配置:"
echo "  GPU数量: $NUM_GPUS"
echo "  每GPU批次大小: $BATCH_SIZE"
echo "  梯度累积步数: $GRAD_ACCUM"
echo "  有效批次大小: $EFFECTIVE_BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 确认是否继续
read -p "是否开始训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

echo "🚀 启动分布式训练..."

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    qwen_finetune_distributed.py \
    --data_path training_data.json \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --logging_steps 5 \
    --save_steps 250 \
    --eval_steps 250

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 分布式训练完成！"
    echo "📁 模型已保存到: $OUTPUT_DIR"
    echo ""
    echo "🧪 测试模型:"
    echo "python inference.py --model_path $OUTPUT_DIR --chapters_text 'your_test_text'"
else
    echo ""
    echo "❌ 训练失败！请检查错误信息。"
fi