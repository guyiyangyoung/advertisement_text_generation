#!/bin/bash

echo "🚀 Quick Start: Qwen-3 8B Fine-tuning on 8 GPUs"
echo "==============================================="

# Optimized settings for 8 GPUs
BATCH_SIZE=6          # Per GPU batch size
GRAD_ACCUM=1          # Gradient accumulation steps
EPOCHS=3              # Number of epochs
LR=5e-5              # Learning rate

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((8 * BATCH_SIZE * GRAD_ACCUM))

echo "🔧 Optimized Configuration for 8 GPUs:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Total epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

if [ ! -f "clean_data.csv" ]; then
    echo "❌ clean_data.csv not found. Please place your data file in the current directory."
    exit 1
fi

if [ ! -f "training_data.json" ]; then
    echo "📊 Preprocessing data..."
    python data_preprocessing.py
    if [ $? -ne 0 ]; then
        echo "❌ Data preprocessing failed!"
        exit 1
    fi
    echo "✅ Data preprocessing completed"
else
    echo "✅ training_data.json found"
fi

# Check GPU availability
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
if [ "$NUM_GPUS" != "8" ]; then
    echo "⚠️  Expected 8 GPUs, found $NUM_GPUS"
    echo "Proceeding with $NUM_GPUS GPUs..."
fi

echo "📊 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

echo ""
echo "🚀 Starting distributed training..."
echo "This will take several hours depending on your data size."
echo ""

# Start training with optimal settings
./run_distributed_training.sh \
    --strategy auto \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --output_dir "./qwen_advertising_copy_8gpu"

echo ""
echo "🎉 Training completed!"
echo "📁 Model saved to: ./qwen_advertising_copy_8gpu"
echo ""
echo "🧪 Test your model:"
echo "python inference.py --model_path ./qwen_advertising_copy_8gpu --chapters_text 'your_test_text'"