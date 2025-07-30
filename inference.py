import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
import logging
from typing import List, Dict
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenAdvertisingCopyGenerator:
    """Generator for advertising copy using fine-tuned Qwen model"""
    
    def __init__(self, 
                 model_path: str,
                 base_model_name: str = "Qwen/Qwen2.5-8B-Instruct",
                 use_quantization: bool = True):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Setup quantization config if needed
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.use_quantization else None
        )
        
        # Load LoRA weights if they exist
        adapter_path = os.path.join(self.model_path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            logger.info("Loading LoRA adapter weights...")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        else:
            logger.warning("No LoRA adapter found, using base model")
        
        logger.info("Model loading complete")
    
    def create_prompt(self, chapters_text: str) -> str:
        """Create a prompt for advertising copy generation"""
        prompt = f"请根据以下章节内容生成优秀的广告文案\n\n{chapters_text}"
        return prompt
    
    def format_conversation(self, prompt: str) -> str:
        """Format prompt into conversation format for Qwen"""
        conversation = f"<|im_start|>system\n你是一个专业的广告文案创作助手，擅长根据详细的内容创作吸引人的广告文案。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return conversation
    
    def generate_advertising_copy(self, 
                                chapters_text: str,
                                max_length: int = 1024,
                                temperature: float = 0.7,
                                top_p: float = 0.9,
                                do_sample: bool = True,
                                repetition_penalty: float = 1.1) -> str:
        """Generate advertising copy from chapters text"""
        
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Create prompt
        prompt = self.create_prompt(chapters_text)
        
        # Format conversation
        conversation = self.format_conversation(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Clean up response (remove any system tokens that might leak)
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        return response
    
    def batch_generate(self, 
                      chapters_list: List[str],
                      max_length: int = 1024,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      do_sample: bool = True,
                      repetition_penalty: float = 1.1) -> List[str]:
        """Generate advertising copy for multiple chapters"""
        
        results = []
        for i, chapters_text in enumerate(chapters_list):
            logger.info(f"Generating copy for example {i+1}/{len(chapters_list)}")
            
            try:
                copy = self.generate_advertising_copy(
                    chapters_text=chapters_text,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty
                )
                results.append(copy)
            except Exception as e:
                logger.error(f"Error generating copy for example {i+1}: {e}")
                results.append("")
        
        return results

def parse_chapters_text(chapters_text: str) -> str:
    """Parse and format chapters text from JSON string"""
    try:
        if isinstance(chapters_text, str):
            try:
                chapters = json.loads(chapters_text)
            except json.JSONDecodeError:
                import ast
                chapters = ast.literal_eval(chapters_text)
        else:
            chapters = chapters_text
        
        if not isinstance(chapters, list):
            return chapters_text
        
        formatted_text = ""
        for chapter in chapters:
            if isinstance(chapter, dict):
                title = chapter.get('title', '')
                content = chapter.get('content', '')
                formatted_text += f"章节标题: {title}\n章节内容: {content}\n\n"
        
        return formatted_text.strip()
    
    except Exception as e:
        logger.error(f"Error parsing chapters: {e}")
        return str(chapters_text)

def main():
    parser = argparse.ArgumentParser(description="Generate advertising copy using fine-tuned Qwen model")
    parser.add_argument("--model_path", type=str, default="./qwen_advertising_copy", 
                       help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-8B-Instruct",
                       help="Base model name")
    parser.add_argument("--chapters_text", type=str, 
                       help="Chapters text (JSON string or formatted text)")
    parser.add_argument("--input_file", type=str,
                       help="Input file containing chapters text")
    parser.add_argument("--output_file", type=str,
                       help="Output file to save generated copy")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for generation")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = QwenAdvertisingCopyGenerator(
        model_path=args.model_path,
        base_model_name=args.base_model,
        use_quantization=not args.no_quantization
    )
    
    # Load model
    generator.load_model()
    
    # Generate copy
    if args.chapters_text:
        # Single generation
        formatted_chapters = parse_chapters_text(args.chapters_text)
        copy = generator.generate_advertising_copy(
            chapters_text=formatted_chapters,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print("Generated Advertising Copy:")
        print("=" * 50)
        print(copy)
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(copy)
            print(f"\nSaved to {args.output_file}")
    
    elif args.input_file:
        # Batch generation from file
        logger.info(f"Loading chapters from {args.input_file}")
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    chapters_list = [parse_chapters_text(item) for item in data]
                else:
                    chapters_list = [parse_chapters_text(data)]
            else:
                # Assume text file with one chapters text per line
                chapters_list = [parse_chapters_text(line.strip()) for line in f if line.strip()]
        
        # Generate copies
        copies = generator.batch_generate(
            chapters_list=chapters_list,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Save results
        output_file = args.output_file or "generated_copies.json"
        results = [{"chapters": chapters, "advertising_copy": copy} 
                  for chapters, copy in zip(chapters_list, copies)]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(copies)} advertising copies")
        print(f"Results saved to {output_file}")
    
    else:
        print("Please provide either --chapters_text or --input_file")

if __name__ == "__main__":
    main()