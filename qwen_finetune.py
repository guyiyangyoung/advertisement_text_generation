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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenFineTuner:
    """Fine-tuner for Qwen-3 8B model for advertising copy generation"""
    
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
        
        # Setup data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        logger.info("Model and tokenizer setup complete")
    
    def format_conversation(self, instruction: str, output: str) -> str:
        """Format instruction and output into conversation format for Qwen"""
        # Qwen conversation format
        conversation = f"<|im_start|>system\n你是一个专业的广告文案创作助手，擅长根据详细的内容创作吸引人的广告文案。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return conversation
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize the examples for training"""
        # Format conversations
        conversations = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            conversation = self.format_conversation(instruction, output)
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
                                output_dir: str = "./qwen_advertising_copy",
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 4,
                                per_device_eval_batch_size: int = 4,
                                gradient_accumulation_steps: int = 4,
                                learning_rate: float = 5e-5,
                                warmup_steps: int = 100,
                                logging_steps: int = 10,
                                save_steps: int = 500,
                                eval_steps: int = 500,
                                save_total_limit: int = 3,
                                use_wandb: bool = False):
        """Setup training arguments"""
        
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
            report_to="wandb" if use_wandb else None,
            run_name="qwen-advertising-copy-finetune" if use_wandb else None,
        )
    
    def train(self, 
              data_path: str,
              output_dir: str = "./qwen_advertising_copy",
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
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        
        return trainer

def main():
    """Main function to run fine-tuning"""
    
    # Configuration
    config = {
        "data_path": "training_data.json",
        "output_dir": "./qwen_advertising_copy",
        "model_name": "Qwen/Qwen2.5-8B-Instruct",
        "use_quantization": True,
        "use_lora": True,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "use_wandb": False  # Set to True if you want to use Weights & Biases
    }
    
    # Check if data file exists
    if not os.path.exists(config["data_path"]):
        logger.error(f"Data file {config['data_path']} not found. Please run data_preprocessing.py first.")
        return
    
    # Initialize fine-tuner
    fine_tuner = QwenFineTuner(
        model_name=config["model_name"],
        use_quantization=config["use_quantization"],
        use_lora=config["use_lora"]
    )
    
    # Start training
    trainer = fine_tuner.train(
        data_path=config["data_path"],
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        use_wandb=config["use_wandb"]
    )
    
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()