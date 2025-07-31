# Qwen-3 8B Fine-tuning for Advertising Copy Generation

This repository contains a complete pipeline for fine-tuning the Qwen-3 8B model to generate excellent advertising copy from detailed chapter text.

## Overview

The system takes detailed chapter content (title and text) as input and generates compelling advertising copy that:
- Grabs readers' attention
- Highlights content appeal and attractions
- Stimulates reading interest
- Uses vivid and interesting language

## Features

- **Data Preprocessing**: Handles JSON-formatted chapter data and prepares it for training
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using LoRA adapters
- **4-bit Quantization**: Memory-efficient training with BitsAndBytesConfig
- **Batch Processing**: Support for batch inference
- **Flexible Configuration**: Easy-to-modify training parameters

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- At least 16GB GPU memory for 4-bit quantized training
- 32GB+ system RAM recommended

## Installation

### Quick Setup

Run the setup script to automatically install all dependencies:

```bash
./setup.sh
```

### Manual Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

Your `clean_data.csv` should have two columns:

- `chapters_text`: JSON string containing a list of chapter objects with `title` and `content` fields
- `revise_asr`: The target advertising copy text

Example chapter format:
```json
[
  {
    "title": "第01章 解开你的腰带",
    "content": "01 解开你的腰带\\n"哎呀你别催了，这暴雨天路上堵得厉害，我也着急呀！"..."
  },
  {
    "title": "第02章 跪下",
    "content": "..."
  }
]
```

## Usage

### 1. Data Preprocessing

First, convert your CSV data to training format:

```bash
python data_preprocessing.py
```

This will:
- Parse the `chapters_text` JSON data
- Format chapters into readable text
- Create structured prompts for training
- Save processed data as `training_data.json`

### 2. Fine-tuning

Start the fine-tuning process:

```bash
python qwen_finetune.py
```

**Training Configuration:**
- Model: Qwen/Qwen2.5-8B-Instruct
- LoRA rank: 16
- Learning rate: 5e-5
- Batch size: 2 (with gradient accumulation)
- Epochs: 3
- 4-bit quantization enabled

The model will be saved to `./qwen_advertising_copy/`

### 3. Inference

Generate advertising copy from chapter text:

#### Single Generation
```bash
python inference.py --chapters_text '[{"title": "第01章 标题", "content": "章节内容..."}]'
```

#### Batch Generation
```bash
python inference.py --input_file chapters.json --output_file results.json
```

#### Custom Parameters
```bash
python inference.py \
    --chapters_text "your_text" \
    --model_path "./qwen_advertising_copy" \
    --max_length 512 \
    --temperature 0.7 \
    --top_p 0.9
```

## Configuration

### Training Parameters

You can modify training parameters in `qwen_finetune.py`:

```python
config = {
    "num_train_epochs": 3,           # Number of training epochs
    "per_device_train_batch_size": 2, # Batch size per device
    "gradient_accumulation_steps": 8, # Gradient accumulation
    "learning_rate": 5e-5,           # Learning rate
    "warmup_steps": 100,             # Warmup steps
    "save_steps": 500,               # Model save frequency
    "eval_steps": 500,               # Evaluation frequency
}
```

### LoRA Configuration

LoRA parameters can be adjusted in the `QwenFineTuner` class:

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha
    lora_dropout=0.1,        # LoRA dropout
    target_modules=[...],    # Target modules
)
```

### Generation Parameters

Customize generation behavior:

```python
generator.generate_advertising_copy(
    chapters_text=text,
    max_length=512,          # Maximum output length
    temperature=0.7,         # Sampling temperature
    top_p=0.9,              # Top-p sampling
    repetition_penalty=1.1,  # Repetition penalty
)
```

## File Structure

```
.
├── data_preprocessing.py    # Data preprocessing script
├── qwen_finetune.py        # Main fine-tuning script
├── inference.py            # Inference and generation script
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── README.md              # This file
├── clean_data.csv         # Your input data (to be provided)
├── training_data.json     # Processed training data
└── qwen_advertising_copy/ # Output model directory
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── tokenizer.json
    └── ...
```

## Monitoring Training

### Option 1: Console Output
Monitor training progress through console logs showing loss, learning rate, and evaluation metrics.

### Option 2: Weights & Biases
Enable W&B logging by setting `use_wandb=True` in the config:

```python
config["use_wandb"] = True
```

Then login to W&B:
```bash
wandb login
```

## Memory Requirements

### Training
- **4-bit Quantized + LoRA**: ~12-16GB GPU memory
- **Full Precision**: ~32GB+ GPU memory
- **System RAM**: 16GB+ recommended

### Inference
- **4-bit Quantized**: ~8-12GB GPU memory
- **Full Precision**: ~16GB+ GPU memory

## Troubleshooting

### Out of Memory
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing
4. Use smaller LoRA rank

### Slow Training
1. Enable mixed precision training (fp16)
2. Use gradient checkpointing
3. Optimize data loading with more workers

### Generation Quality
1. Adjust temperature (lower = more focused)
2. Modify top_p (lower = more conservative)
3. Increase max_length for longer outputs
4. Train for more epochs

## Example Workflow

1. **Setup Environment**:
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. **Prepare Data**:
   ```bash
   # Place your clean_data.csv in the current directory
   python data_preprocessing.py
   ```

3. **Train Model**:
   ```bash
   python qwen_finetune.py
   ```

4. **Generate Copy**:
   ```bash
   python inference.py --chapters_text '[{"title": "测试章节", "content": "这是一个测试章节的内容..."}]'
   ```

## License

This project uses the Qwen model which has its own license terms. Please refer to the official Qwen documentation for licensing information.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this fine-tuning pipeline.