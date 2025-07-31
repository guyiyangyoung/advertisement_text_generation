import pandas as pd
import json
import ast
from typing import List, Dict, Any
from datasets import Dataset
import os

class DataPreprocessor:
    """Preprocessor for advertising copy generation training data"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV data"""
        print(f"Loading data from {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} rows")
        return self.data
    
    def parse_chapters_text(self, chapters_text: str) -> List[Dict[str, str]]:
        """Parse chapters_text from string representation to list of dicts"""
        try:
            # Handle if it's already a string representation of a list
            if isinstance(chapters_text, str):
                # Try to parse as JSON first
                try:
                    chapters = json.loads(chapters_text)
                except json.JSONDecodeError:
                    # If JSON fails, try ast.literal_eval
                    chapters = ast.literal_eval(chapters_text)
            else:
                chapters = chapters_text
                
            return chapters if isinstance(chapters, list) else []
        except Exception as e:
            print(f"Error parsing chapters: {e}")
            return []
    
    def format_chapters_as_text(self, chapters: List[Dict[str, str]]) -> str:
        """Convert chapters list to formatted text"""
        if not chapters:
            return ""
            
        formatted_text = ""
        for chapter in chapters:
            title = chapter.get('title', '')
            content = chapter.get('content', '')
            formatted_text += f"章节标题: {title}\n章节内容: {content}\n\n"
        
        return formatted_text.strip()
    
    def create_training_prompt(self, chapters_text: str) -> str:
        """Create a structured prompt for training"""
        prompt = f"""请根据以下详细的章节内容，生成一段优秀的广告文案。广告文案应该：
1. 抓住读者的注意力
2. 突出内容的亮点和吸引力
3. 激发读者的阅读兴趣
4. 语言生动有趣

章节详情：
{chapters_text}

广告文案："""
        return prompt
    
    def prepare_training_data(self) -> List[Dict[str, str]]:
        """Prepare data in the format needed for fine-tuning"""
        if self.data is None:
            self.load_data()
            
        training_data = []
        
        for idx, row in self.data.iterrows():
            # Parse chapters text
            chapters = self.parse_chapters_text(row['chapters_text'])
            
            if not chapters:
                print(f"Skipping row {idx}: Invalid chapters_text")
                continue
                
            # Format chapters as readable text
            formatted_chapters = self.format_chapters_as_text(chapters)
            
            if not formatted_chapters or pd.isna(row['revise_asr']):
                print(f"Skipping row {idx}: Missing data")
                continue
            
            # Create training prompt
            input_text = self.create_training_prompt(formatted_chapters)
            
            # Create training example
            training_example = {
                "instruction": input_text,
                "output": str(row['revise_asr']).strip(),
                "input": ""  # Empty input as we include everything in instruction
            }
            
            training_data.append(training_example)
            
            if len(training_data) % 100 == 0:
                print(f"Processed {len(training_data)} examples")
        
        print(f"Prepared {len(training_data)} training examples")
        return training_data
    
    def save_training_data(self, training_data: List[Dict[str, str]], output_path: str):
        """Save training data as JSON for fine-tuning"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"Saved training data to {output_path}")
    
    def create_huggingface_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """Create HuggingFace Dataset from training data"""
        return Dataset.from_list(training_data)

def main():
    # Configuration
    csv_path = "clean_data.csv"  # Update this path as needed
    output_json_path = "training_data.json"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please ensure the file exists in the current directory.")
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(csv_path)
    
    # Prepare training data
    training_data = preprocessor.prepare_training_data()
    
    if not training_data:
        print("No training data prepared. Please check your CSV file format.")
        return
    
    # Save training data
    preprocessor.save_training_data(training_data, output_json_path)
    
    # Create HuggingFace dataset
    dataset = preprocessor.create_huggingface_dataset(training_data)
    print(f"Created HuggingFace dataset with {len(dataset)} examples")
    
    # Display sample
    print("\nSample training example:")
    print("=" * 50)
    print(f"Instruction: {training_data[0]['instruction'][:200]}...")
    print(f"Output: {training_data[0]['output'][:100]}...")

if __name__ == "__main__":
    main()