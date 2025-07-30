#!/bin/bash

echo "🚀 Qwen-3 8B Single GPU Fine-tuning"
echo "==================================="

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

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected, will use CPU (very slow)"
fi

# Optimized settings for single GPU
BATCH_SIZE=2          # Batch size (adjust based on GPU memory)
GRAD_ACCUM=8          # Gradient accumulation steps
EPOCHS=3              # Number of epochs
LR=5e-5              # Learning rate

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

echo ""
echo "🔧 Training Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Quantization: Enabled (4-bit)"
echo "  LoRA: Enabled"
echo ""

echo "🚀 Starting single GPU training..."
echo "This may take several hours depending on your data size and GPU."
echo ""

# Start training
python train_single_gpu.py \
    --data_path training_data.json \
    --output_dir ./qwen_advertising_copy_single \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --learning_rate $LR

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model saved to: ./qwen_advertising_copy_single"
    echo ""
    echo "🧪 Test your model:"
    echo "python inference.py --model_path ./qwen_advertising_copy_single --chapters_text 'your_test_text'"
else
    echo ""
    echo "❌ Training failed! Check the error messages above."
fi