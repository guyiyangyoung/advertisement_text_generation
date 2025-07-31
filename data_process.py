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
        # Simple, direct instruction that focuses on the task
        prompt = f"请根据以下章节内容生成优秀的广告文案：\n\n{chapters_text}"
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
            
            # Create training example with clear separation
            training_example = {
                "instruction": 
                        """
                            ## 角色:
                            你是一名资深纯小说内容口播文案的策划，擅长精准捕捉小说核心卖点，能用口语化、有感染力的表达激发读者阅读欲，尤其擅长适配不同题材的语言风格，精通听觉化表达设计（如语气调控、短句优化），能根据具体题材调整叙事节奏和情感侧重。

                            ## 目标:
                            - 字数控制：500-1000 字，符合小说口播节奏
                            - 开头设计：用小说核心冲突场景直接开场，30秒内必须包含强冲突+明确情绪标注
                            - 中间叙事：按 “核心冲突→关键转折→爽点 / 情感爆点” 逻辑推进，避免复杂人名/设定堆砌，突出 “主角如何破局”“关系如何变化” 等核心看点
                            - 结尾钩子：停在角色具体动作/关键台词/心理活动

                            ## 内容审查要求：
                            - 规避敏感内容：
                            - 不涉及政治相关（含人物、事件、象征符号等）；
                            - 政府机关名称替换为生活化表述（如 “公安局”→“安保办公室”，“民政局”→“登记大厅”）；
                            - 无任何色情描述（含身体暗示、低俗互动）；
                            - 无暴力细节（规避鲜血、殴打等直接描写，冲突可用 “争执”“对峙”“气场压制” 等中性词）。

                            ## 输入内容差异化处理策略：
                            - 若输入为原始章节片段：优先提取 “场景画面感”“对话冲突”“即时情绪爆发点”，弱化次要剧情；
                            - 若输入为已有广告文案：保留核心卖点，优化口语化表达。

                            ## 创作流程：
                            1. 题材定位：先明确小说具体题材（从上述题材中匹配，或补充其他细分题材），锁定对应目标人群（如言情→年轻女性读者，悬疑→推理爱好者）；
                            2. 核心提取：从输入内容中抓 3 个关键要素 —— 主角核心困境（与题材强相关）、最大爽点 / 泪点（贴合题材风格）、关键转折点（推动题材主线）；
                            3. 结构搭建：按 “情绪引爆开头→冲突升级中间→钩子结尾” 梳理剧情线，确保每 200 字有一个小高潮，语言风格严格遵循对应题材的表达偏好；
                            4. 风格适配：根据题材调整用词（如言情多用情感词，悬疑多用氛围词，玄幻多用爽感词），避免跨题材风格混乱；
                            5. 合规润色：检查敏感词，优化表述流畅度，重点检查开头结尾禁令，确保风格与题材高度统一。

                            ## 强制禁止：
                            - 开头禁令：禁用“家人们/宝子们/今天推荐/今天给大家讲/这本小说是”等主播式引导语
                            - 结尾禁令：禁用“接下来会怎样”“该如何应对”等说书人提问句式
                            - 全局禁令：禁用 '[]'、'【】' 等特殊格式，绝对禁止解说词（禁用 "这个故事讲的是"" 接下来 "," 这是 xx 的故事 ""这是一个... 故事"" 这类故事适合... 读者 " 等引导语）

                            请根据以下详细的章节内容，生成一段符合上述要求的文案：

                            章节详情：
                        """,
                "input": formatted_chapters,  # The chapters_text is the actual input
                "output": str(row['revise_asr']).strip()  # The revise_asr is the target output
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
    csv_path = "/mnt/bn/ug-diffusion-lq/guyiyang/clean_data.csv"
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