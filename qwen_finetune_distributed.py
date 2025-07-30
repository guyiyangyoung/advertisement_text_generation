import os
import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb
from typing import Dict, List, Optional
import logging
import torch.distributed as dist
from accelerate import Accelerator
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedQwenFineTuner:
    """Distributed Fine-tuner for Qwen-3 8B model using multiple GPUs"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-8B-Instruct",
                 use_quantization: bool = True,
                 use_lora: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with quantization if specified"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Setup quantization config if needed
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model with device_map="auto" for multi-GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_quantization else None
        )
        
        # Setup LoRA if specified
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                self.model.print_trainable_parameters()
        
        # Setup data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        logger.info("Model and tokenizer setup complete")
    
    def format_conversation(self, instruction: str, input_text: str, output: str) -> str:
        """Format instruction, input and output into conversation format for Qwen"""
        # Combine instruction and input for the user message
        if input_text.strip():
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
            
        # Qwen conversation format
        conversation = f"<|im_start|>system\n你是一个专业的广告文案创作助手，擅长根据详细的内容创作吸引人的广告文案。<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return conversation
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize the examples for training"""
        # Format conversations
        conversations = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            conversation = self.format_conversation(instruction, input_text, output)
            conversations.append(conversation)
        
        # Tokenize
        model_inputs = self.tokenizer(
            conversations,
            max_length=2048,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Set labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def load_and_prepare_dataset(self, data_path: str, test_size: float = 0.1):
        """Load and prepare dataset for training"""
        logger.info(f"Loading dataset from {data_path}")
        
        # Load the JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Split into train and validation
        dataset = dataset.train_test_split(test_size=test_size, seed=42)
        
        # Tokenize datasets
        train_dataset = dataset["train"].map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing train dataset"
        )
        
        eval_dataset = dataset["test"].map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["test"].column_names,
            desc="Tokenizing eval dataset"
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def setup_training_arguments(self, 
                                output_dir: str = "./qwen_advertising_copy_distributed",
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 4,
                                per_device_eval_batch_size: int = 4,
                                gradient_accumulation_steps: int = 2,
                                learning_rate: float = 5e-5,
                                warmup_steps: int = 100,
                                logging_steps: int = 10,
                                save_steps: int = 500,
                                eval_steps: int = 500,
                                save_total_limit: int = 3,
                                use_wandb: bool = False):
        """Setup training arguments for distributed training"""
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            # Distributed training settings
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            group_by_length=True,
            length_column_name="length",
            # Logging and reporting
            report_to="wandb" if use_wandb else None,
            run_name="qwen-advertising-copy-distributed" if use_wandb else None,
            # Performance optimizations
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            save_safetensors=True,
        )
    
    def train(self, 
              data_path: str,
              output_dir: str = "./qwen_advertising_copy_distributed",
              **training_kwargs):
        """Main training function"""
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load and prepare dataset
        train_dataset, eval_dataset = self.load_and_prepare_dataset(data_path)
        
        # Setup training arguments
        training_args = self.setup_training_arguments(output_dir=output_dir, **training_kwargs)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        # Start training
        logger.info("Starting distributed training...")
        trainer.train()
        
        # Save the final model (only on main process)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info("Saving final model...")
            trainer.save_model()
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Training complete! Model saved to {output_dir}")
        
        return trainer

def main():
    """Main function for distributed training"""
    
    parser = argparse.ArgumentParser(description="Distributed fine-tuning of Qwen-3 8B")
    parser.add_argument("--data_path", type=str, default="training_data.json",
                       help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default="./qwen_advertising_copy_distributed",
                       help="Output directory for the model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-8B-Instruct",
                       help="Model name or path")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file {args.data_path} not found. Please run data_preprocessing.py first.")
        return
    
    # Log GPU information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("No CUDA GPUs found")
    
    # Initialize fine-tuner
    fine_tuner = DistributedQwenFineTuner(
        model_name=args.model_name,
        use_quantization=not args.no_quantization,
        use_lora=not args.no_lora
    )
    
    # Start training
    trainer = fine_tuner.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=args.use_wandb
    )
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info("Distributed fine-tuning completed successfully!")

if __name__ == "__main__":
    main()