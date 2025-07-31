import os
import torch
import pandas as pd
import json
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import PeftModel
import argparse
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenInference:
    """Qwen model inference class with GPU support"""
    
    def __init__(self, 
                 model_path: str = "/mnt/bn/ug-diffusion-lq/guyiyang/qwen_advertising_copy",
                 base_model_path: str = "/mnt/bn/ug-diffusion-lq/guyiyang/Qwen3-8B",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True):
        
        self.model_path = model_path
        self.base_model_path = base_model_path
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
        
        # Load base model
        logger.info(f"Loading base model from {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto" if self.device_count > 1 else "cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load fine-tuned weights
        logger.info(f"Loading fine-tuned weights from {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Merge weights for faster inference
        self.model = self.model.merge_and_unload()
        
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Print memory usage
        if torch.cuda.is_available():
            for i in range(self.device_count):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i} Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    def generate_single(self, text):
        """Generate output for a single input text"""
        # Format the input according to training format
        user_message = f"请根据以下小说章节内容，生成一段精彩的口播文案，要求语言生动、有感染力，能够激发读者的阅读兴趣。\n\n{text}"
        
        conversation = f"<|im_start|>system\n你是一名资深纯小说内容口播文案的策划，擅长精准捕捉小说核心卖点，能用口语化、有感染力的表达激发读者阅读欲，尤其擅长适配不同题材的语言风格，精通听觉化表达设计（如语气调控、短句优化），能根据具体题材调整叙事节奏和情感侧重。<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            conversation,
            truncation=True,
            max_length=4096,
            padding=False,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        prompt_length = inputs['input_ids'].shape[1]
        
        # Debug: Print prompt info
        logger.debug(f"Prompt length: {prompt_length}")
        logger.debug(f"Prompt text: {conversation[:200]}...")
        
        with torch.no_grad():
            # Generate outputs
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
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
        
        # Debug: Print generation info
        total_length = outputs[0].shape[0]
        logger.debug(f"Total generated length: {total_length}")
        logger.debug(f"New tokens generated: {total_length - prompt_length}")
        
        # Decode the full output first to debug
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Full output: {full_output[:500]}...")
        
        # Extract only the generated part (after the prompt)
        if total_length > prompt_length:
            generated_tokens = outputs[0][prompt_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Further clean up: remove any remaining prompt artifacts
            # Look for the assistant response start
            if "<|im_start|>assistant\n" in full_output:
                assistant_start = full_output.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                generated_text = full_output[assistant_start:]
            
            # Remove any end tokens
            if "<|im_end|>" in generated_text:
                generated_text = generated_text.split("<|im_end|>")[0]
            
            # Comprehensive cleaning of generated text
            
            # 1. Remove <think> tags and their content (case insensitive)
            generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL | re.IGNORECASE)
            
            # 2. Remove genre/style tags like 【穿越重生+女扮男装+先知预言+权谋斗争】
            generated_text = re.sub(r'【[^】]*】', '', generated_text)
            
            # 3. Remove square bracket tags like [标签内容]
            generated_text = re.sub(r'\[[^\]]*\]', '', generated_text)
            
            # 4. Remove any other HTML-like tags
            generated_text = re.sub(r'<[^>]*>', '', generated_text)
            
            # 5. Remove common unwanted phrases/patterns at the beginning
            unwanted_patterns = [
                r'^.*?(?=[\u4e00-\u9fff])',  # Remove anything before first Chinese character
                r'^[+\-=*]*',  # Remove leading symbols
                r'^\s*类型[:：].*?\n',  # Remove type labels
                r'^\s*风格[:：].*?\n',  # Remove style labels
            ]
            
            for pattern in unwanted_patterns:
                generated_text = re.sub(pattern, '', generated_text, flags=re.MULTILINE)
            
            # 6. Clean up whitespace
            generated_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', generated_text)  # Multiple newlines to double
            generated_text = re.sub(r'^\s+|\s+$', '', generated_text)  # Trim whitespace
            generated_text = re.sub(r' +', ' ', generated_text)  # Multiple spaces to single
            
            # 7. Remove any remaining empty lines at the start
            generated_text = generated_text.lstrip('\n ')
            
            logger.debug(f"Extracted generated text: {generated_text[:200]}...")
            return generated_text.strip()
        else:
            logger.warning("No new tokens were generated!")
            return "Error: No new content generated"
    
    def infer_from_csv(self, csv_path: str, output_path: str):
        """Run inference on test.csv and save results"""
        logger.info(f"Loading data from {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Extract chapters_text column
        if 'chapters_text' not in df.columns:
            raise ValueError("'chapters_text' column not found in CSV")
        
        texts = df['chapters_text'].fillna("").astype(str).tolist()
        
        # Run inference for each text
        all_results = []
        generated_outputs = []
        start_time = time.time()
        
        logger.info("Starting inference...")
        
        for i, text in enumerate(tqdm(texts, desc="Generating")):
            try:
                if text.strip():  # Only process non-empty texts
                    generated_text = self.generate_single(text)
                    generated_outputs.append(generated_text)
                    
                    all_results.append({
                        'input_text': text,
                        'generated_output': generated_text
                    })
                else:
                    generated_outputs.append("Error: Empty input text")
                    all_results.append({
                        'input_text': text,
                        'generated_output': "Error: Empty input text"
                    })
                
                # Log progress every 10 samples
                if (i + 1) % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_sample = elapsed_time / (i + 1)
                    remaining_samples = len(texts) - (i + 1)
                    estimated_remaining_time = avg_time_per_sample * remaining_samples
                    logger.info(f"Processed {i + 1}/{len(texts)} samples, "
                              f"avg time per sample: {avg_time_per_sample:.2f}s, "
                              f"estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                generated_outputs.append(f"Error: {str(e)}")
                all_results.append({
                    'input_text': text,
                    'generated_output': f"Error: {str(e)}"
                })
        
        # Create output dataframe
        output_df = df.copy()
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
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for nucleus sampling")

    parser.add_argument("--no_sample", action="store_true",
                       help="Disable sampling (use greedy decoding)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, will use CPU (very slow)")
    else:
        logger.info(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
    
    # Initialize inference class
    inference = QwenInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
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
        output_path=args.output_path
    )
    
    logger.info("Inference completed successfully!")
    logger.info(f"Generated outputs for {len(results)} samples")

def test_single_generation():
    """Test function to verify generation works correctly"""
    logger.info("Running single generation test...")
    
    # Initialize inference class
    inference = QwenInference()
    
    # Load model
    inference.load_model()
    
    # Test with a simple text
    test_text = "沈从妩觉得自己简直是个大冤种，好不容易把家里的跋扈二世祖调教成状元郎，自己也封了诰命，正要走上人生巅峰，怎么一睁眼就回到了解放前？"
    
    result = inference.generate_single(test_text)
    
    logger.info(f"Test input: {test_text[:100]}...")
    logger.info(f"Test output: {result}")
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_generation()
    else:
        main()