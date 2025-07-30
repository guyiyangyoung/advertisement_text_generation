#!/bin/bash

echo "🚀 Qwen-3 8B Distributed Fine-tuning Script"
echo "=========================================="

# Check if we have the required files
if [ ! -f "training_data.json" ]; then
    echo "❌ training_data.json not found. Please run data_preprocessing.py first."
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. CUDA seems not available."
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "🔍 Found $NUM_GPUS GPU(s)"

# Check memory
echo "📊 GPU Memory Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Default configuration for 8 GPUs
DEFAULT_BATCH_SIZE=6
DEFAULT_GRAD_ACCUM=1
DEFAULT_EPOCHS=3
DEFAULT_LR=5e-5

# Parse command line arguments
STRATEGY="ddp"  # Default strategy
BATCH_SIZE=$DEFAULT_BATCH_SIZE
GRAD_ACCUM=$DEFAULT_GRAD_ACCUM
EPOCHS=$DEFAULT_EPOCHS
LEARNING_RATE=$DEFAULT_LR
USE_WANDB=""
OUTPUT_DIR="./qwen_advertising_copy_distributed"

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "🔧 Training Configuration:"
echo "  Strategy: $STRATEGY"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Effective batch size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"

# Create output directory
mkdir -p "$OUTPUT_DIR"

case $STRATEGY in
    "ddp"|"fsdp")
        echo "🚀 Starting distributed training using $STRATEGY..."
        
        if [ "$STRATEGY" = "ddp" ]; then
            # Data Parallel training
            echo "📡 Using DistributedDataParallel (DDP)"
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
                --eval_steps 250 \
                $USE_WANDB
        else
            # Fully Sharded Data Parallel
            echo "🔀 Using Fully Sharded Data Parallel (FSDP)"
            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_port=29500 \
                qwen_finetune_fsdp.py \
                --data_path training_data.json \
                --output_dir "$OUTPUT_DIR" \
                --per_device_train_batch_size $BATCH_SIZE \
                --gradient_accumulation_steps $GRAD_ACCUM \
                --num_train_epochs $EPOCHS \
                --learning_rate $LEARNING_RATE \
                $USE_WANDB
        fi
        ;;
    "deepspeed")
        echo "🚀 Starting training using DeepSpeed..."
        echo "⚡ Using DeepSpeed ZeRO for memory optimization"
        
        # Create DeepSpeed config
        cat > deepspeed_config.json << EOF
{
    "train_batch_size": $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM)),
    "train_micro_batch_size_per_gpu": $BATCH_SIZE,
    "gradient_accumulation_steps": $GRAD_ACCUM,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": $LEARNING_RATE,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": $LEARNING_RATE,
            "warmup_num_steps": 100
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "wall_clock_breakdown": false
}
EOF
        
        deepspeed \
            --num_gpus=$NUM_GPUS \
            qwen_finetune_deepspeed.py \
            --deepspeed deepspeed_config.json \
            --data_path training_data.json \
            --output_dir "$OUTPUT_DIR" \
            --num_train_epochs $EPOCHS \
            $USE_WANDB
        ;;
    "auto")
        echo "🤖 Auto-selecting best strategy for your hardware..."
        
        # Check total GPU memory
        TOTAL_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
        echo "📊 Total GPU Memory: ${TOTAL_MEMORY}MB"
        
        if [ $TOTAL_MEMORY -gt 320000 ]; then  # > 320GB total
            echo "🔀 High memory setup detected, using FSDP"
            STRATEGY="fsdp"
        elif [ $TOTAL_MEMORY -gt 160000 ]; then  # > 160GB total
            echo "📡 Medium memory setup detected, using DDP"
            STRATEGY="ddp"
        else
            echo "⚡ Lower memory setup detected, using DeepSpeed"
            STRATEGY="deepspeed"
        fi
        
        # Recursive call with selected strategy
        bash "$0" --strategy $STRATEGY --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM --epochs $EPOCHS --lr $LEARNING_RATE --output_dir "$OUTPUT_DIR" $USE_WANDB
        ;;
    *)
        echo "❌ Unknown strategy: $STRATEGY"
        echo "Available strategies: ddp, fsdp, deepspeed, auto"
        exit 1
        ;;
esac

echo "✅ Training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo "🎯 You can now use the model for inference with: python inference.py --model_path $OUTPUT_DIR"