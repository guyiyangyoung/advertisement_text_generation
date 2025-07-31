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
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleGPUQwenFineTuner:
    """Single GPU Fine-tuner for Qwen-3 8B model"""
    
    def __init__(self, 
                 model_name: str = "/mnt/bn/ug-diffusion-lq/guyiyang/Qwen3-8B",
                 use_quantization: bool = True,
                 use_lora: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
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
            logger.info("Using 4-bit quantization for memory efficiency")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
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
            self.model.print_trainable_parameters()
            logger.info("LoRA adapters added for efficient fine-tuning")
        
        # Setup data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        logger.info("Model and tokenizer setup complete")
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    def format_conversation(self, instruction: str, input_text: str, output: str) -> str:
        """Format instruction, input and output into conversation format for Qwen"""
        if input_text.strip():
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
            
        conversation = f"<|im_start|>system\n你是一名资深纯小说内容口播文案的策划，擅长精准捕捉小说核心卖点，能用口语化、有感染力的表达激发读者阅读欲，尤其擅长适配不同题材的语言风格，精通听觉化表达设计（如语气调控、短句优化），能根据具体题材调整叙事节奏和情感侧重。<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return conversation
    
    def tokenize_function(self, examples: dict) -> dict:
        """Tokenize the examples for training"""
        conversations = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            conversation = self.format_conversation(instruction, input_text, output)
            conversations.append(conversation)
        
        # Tokenize
        model_inputs = self.tokenizer(
            conversations,
            max_length=4096,
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
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = Dataset.from_list(data)
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
    
    def train(self, 
              data_path: str,
              output_dir: str = "./qwen_advertising_copy",
              num_train_epochs: int = 3,
              per_device_train_batch_size: int = 2,
              gradient_accumulation_steps: int = 8,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              save_steps: int = 500,
              eval_steps: int = 500,
              use_wandb: bool = False):
        """Main training function"""
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load and prepare dataset
        train_dataset, eval_dataset = self.load_and_prepare_dataset(data_path)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else None,
            run_name="qwen-advertising-copy-single-gpu" if use_wandb else None,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            save_safetensors=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        # Calculate effective batch size
        effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
        logger.info(f"Effective batch size: {effective_batch_size}")
        
        # Start training
        logger.info("Starting single GPU training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        
        return trainer

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Single GPU fine-tuning of Qwen-3 8B")
    parser.add_argument("--data_path", type=str, default="/mnt/bn/ug-diffusion-lq/guyiyang/training_data.json",
                       help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default="/mnt/bn/ug-diffusion-lq/guyiyang/qwen_advertising_copy",
                       help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.path.exists(args.data_path):
        logger.error(f"Data file {args.data_path} not found. Please run data_preprocessing.py first.")
        return
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("No CUDA GPU found, training on CPU (very slow)")
    
    # Initialize fine-tuner
    fine_tuner = SingleGPUQwenFineTuner(
        use_quantization=not args.no_quantization,
        use_lora=not args.no_lora
    )
    
    # Start training
    trainer = fine_tuner.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )
    
    logger.info("Single GPU fine-tuning completed successfully!")

if __name__ == "__main__":
    main()