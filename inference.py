import os
import torch
import pandas as pd
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel
import argparse
import logging
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for text inference"""
    
    def __init__(self, texts, tokenizer, max_length=4096):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Format the input according to training format
        user_message = f"请根据以下小说章节内容，生成一段精彩的口播文案，要求语言生动、有感染力，能够激发读者的阅读兴趣。\n\n{text}"
        
        conversation = f"<|im_start|>system\n你是一名资深纯小说内容口播文案的策划，擅长精准捕捉小说核心卖点，能用口语化、有感染力的表达激发读者阅读欲，尤其擅长适配不同题材的语言风格，精通听觉化表达设计（如语气调控、短句优化），能根据具体题材调整叙事节奏和情感侧重。<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'original_text': text,
            'prompt_length': len(inputs['input_ids'].squeeze())
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    original_texts = [item['original_text'] for item in batch]
    prompt_lengths = [item['prompt_length'] for item in batch]
    
    # Pad sequences to the same length
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        pad_length = max_len - len(ids)
        # Pad from left for generation
        padded_ids = torch.cat([torch.full((pad_length,), tokenizer.pad_token_id), ids])
        padded_mask = torch.cat([torch.zeros(pad_length), mask])
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'original_texts': original_texts,
        'prompt_lengths': prompt_lengths
    }

class QwenInference:
    """Qwen model inference class with multi-GPU support"""
    
    def __init__(self, 
                 model_path: str = "/mnt/bn/ug-diffusion-lq/guyiyang/qwen_advertising_copy",
                 base_model_path: str = "/mnt/bn/ug-diffusion-lq/guyiyang/Qwen3-8B",
                 use_quantization: bool = True,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True):
        
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.device_count = torch.cuda.device_count()
        
        logger.info(f"Found {self.device_count} GPUs")
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="left"  # Important for generation
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
        
        # Load base model
        logger.info(f"Loading base model from {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=quantization_config,
            device_map="auto" if self.device_count > 1 else "cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_quantization else None
        )
        
        # Load fine-tuned weights
        logger.info(f"Loading fine-tuned weights from {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Merge weights for faster inference
        self.model = self.model.merge_and_unload()
        
        # Use DataParallel for multi-GPU inference if available
        if self.device_count > 1 and not self.use_quantization:
            logger.info(f"Using DataParallel across {self.device_count} GPUs")
            self.model = DataParallel(self.model)
        
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Print memory usage
        if torch.cuda.is_available():
            for i in range(self.device_count):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i} Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    def generate_batch(self, batch):
        """Generate outputs for a batch of inputs"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        if self.device_count > 0:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        with torch.no_grad():
            # Generate outputs
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                num_return_sequences=1
            )
        
        # Decode outputs
        generated_texts = []
        for i, (output, prompt_length) in enumerate(zip(outputs, batch['prompt_lengths'])):
            # Extract only the generated part (after the prompt)
            generated_tokens = output[prompt_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def infer_from_csv(self, csv_path: str, output_path: str, batch_size: int = 4):
        """Run inference on test.csv and save results"""
        logger.info(f"Loading data from {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Extract chapters_text column
        if 'chapters_text' not in df.columns:
            raise ValueError("'chapters_text' column not found in CSV")
        
        texts = df['chapters_text'].fillna("").astype(str).tolist()
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        logger.info(f"Processing {len(valid_texts)} valid texts")
        
        # Create dataset and dataloader
        dataset = TextDataset(valid_texts, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues with CUDA
        )
        
        # Run inference
        all_results = []
        start_time = time.time()
        
        logger.info(f"Starting inference with batch size {batch_size}")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            try:
                generated_texts = self.generate_batch(batch)
                
                # Store results
                for original_text, generated_text in zip(batch['original_texts'], generated_texts):
                    all_results.append({
                        'input_text': original_text,
                        'generated_output': generated_text
                    })
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_batch = elapsed_time / (batch_idx + 1)
                    logger.info(f"Processed {batch_idx + 1} batches, avg time per batch: {avg_time_per_batch:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Add empty results for this batch to maintain alignment
                for original_text in batch['original_texts']:
                    all_results.append({
                        'input_text': original_text,
                        'generated_output': f"Error: {str(e)}"
                    })
        
        # Create output dataframe
        output_df = df.copy()
        
        # Map results back to original dataframe
        generated_outputs = [""] * len(df)
        result_idx = 0
        
        for i in range(len(df)):
            if i in valid_indices:
                if result_idx < len(all_results):
                    generated_outputs[i] = all_results[result_idx]['generated_output']
                    result_idx += 1
                else:
                    generated_outputs[i] = "Error: No result generated"
            else:
                generated_outputs[i] = "Error: Empty input text"
        
        output_df['generated_advertising_copy'] = generated_outputs
        
        # Save results
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        
        total_time = time.time() - start_time
        logger.info(f"Inference completed in {total_time:.2f}s")
        logger.info(f"Results saved to {output_path}")
        
        # Save additional results as JSON for easier analysis
        json_output_path = output_path.replace('.csv', '_results.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed results saved to {json_output_path}")
        
        return output_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen model")
    parser.add_argument("--model_path", type=str, 
                       default="/mnt/bn/ug-diffusion-lq/guyiyang/qwen_advertising_copy",
                       help="Path to the fine-tuned model")
    parser.add_argument("--base_model_path", type=str,
                       default="/mnt/bn/ug-diffusion-lq/guyiyang/Qwen3-8B", 
                       help="Path to the base Qwen model")
    parser.add_argument("--csv_path", type=str, default="test.csv",
                       help="Path to test CSV file")
    parser.add_argument("--output_path", type=str, default="inference_results.csv",
                       help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for nucleus sampling")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    parser.add_argument("--no_sample", action="store_true",
                       help="Disable sampling (use greedy decoding)")
    
    args = parser.parse_args()
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, will use CPU (very slow)")
    else:
        logger.info(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
    
    # Initialize inference class
    inference = QwenInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        use_quantization=not args.no_quantization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample
    )
    
    # Load model
    inference.load_model()
    
    # Run inference
    results = inference.infer_from_csv(
        csv_path=args.csv_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
    
    logger.info("Inference completed successfully!")
    logger.info(f"Generated outputs for {len(results)} samples")

if __name__ == "__main__":
    main()