import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
from typing import List, Dict, Optional

class Qwen3ChapterSummarizer:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        初始化Qwen3章节概括器
        
        Args:
            model_path: 模型路径，可以是：
                - "Qwen/Qwen2.5-3B-Instruct" (公开可用)
                - "/path/to/local/qwen3-8b" (本地模型路径)
                - "Qwen/Qwen2.5-7B-Instruct" (更大的公开模型)
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print("模型加载完成!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用备用模型...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """加载备用模型"""
        fallback_model = "Qwen/Qwen2.5-3B-Instruct"
        print(f"加载备用模型: {fallback_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    def load_and_parse_csv(self, csv_file: str) -> Optional[List[Dict]]:
        """
        加载CSV文件并解析chapters_text列
        """
        print(f"正在读取文件: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"CSV文件列名: {list(df.columns)}")
            
            if 'chapters_text' not in df.columns:
                print("错误: CSV文件中没有找到 'chapters_text' 列")
                return None
            
            # 获取chapters_text列的数据 (假设只有一行数据)
            chapters_text = df['chapters_text'].iloc[0]
            
            # 解析JSON数据
            chapters_data = json.loads(chapters_text)
            print(f"成功解析出 {len(chapters_data)} 个章节")
            
            # 显示前几个章节的标题
            for i, chapter in enumerate(chapters_data[:3]):
                title = chapter.get('title', f'第{i+1}章')
                print(f"  章节 {i+1}: {title}")
            
            return chapters_data
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print("尝试修复JSON格式...")
            return self._try_fix_json(chapters_text)
        except Exception as e:
            print(f"文件读取错误: {e}")
            return None
    
    def _try_fix_json(self, text: str) -> Optional[List[Dict]]:
        """尝试修复JSON格式"""
        try:
            # 移除可能的前后缀
            text = text.strip()
            if not text.startswith('['):
                # 寻找第一个 [
                start = text.find('[')
                if start != -1:
                    text = text[start:]
            
            if not text.endswith(']'):
                # 寻找最后一个 ]
                end = text.rfind(']')
                if end != -1:
                    text = text[:end+1]
            
            return json.loads(text)
        except:
            print("JSON修复失败")
            return None
    
    def clean_text(self, text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""
        
        # 移除换行符和多余空格
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊控制字符，但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\u0030-\u0039\u0041-\u005a\u0061-\u007a\s，。！？；：""''（）《》【】、]', '', text)
        
        return text.strip()
    
    def create_summary_prompt(self, title: str, content: str) -> str:
        """创建概括提示词"""
        return f"""请将以下小说章节概括为200字左右的精炼文字。要求：
1. 突出关键情节和人物动作
2. 保持故事连贯性
3. 语言简洁明了
4. 字数控制在180-220字之间

章节标题：{title}

章节内容：
{content}

概括："""
    
    def summarize_chapter(self, title: str, content: str, max_length: int = 2000) -> str:
        """
        使用模型对单个章节进行概括
        """
        # 清理内容
        cleaned_content = self.clean_text(content)
        
        # 限制输入长度
        if len(cleaned_content) > max_length:
            cleaned_content = cleaned_content[:max_length] + "..."
        
        # 创建提示词
        prompt = self.create_summary_prompt(title, cleaned_content)
        
        try:
            # 构建对话
            messages = [
                {"role": "system", "content": "你是一个专业的小说内容概括助手，擅长提取关键情节，用简洁的语言概括章节内容。"},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.3,  # 降低随机性，使输出更稳定
                    top_p=0.8,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"生成概括时出错: {e}")
            return f"[生成失败] {str(e)}"
    
    def process_all_chapters(self, chapters_data: List[Dict]) -> List[Dict]:
        """处理所有章节"""
        results = []
        
        print(f"开始处理 {len(chapters_data)} 个章节...")
        
        for i, chapter in enumerate(chapters_data, 1):
            title = chapter.get('title', f'第{i}章')
            content = chapter.get('content', '')
            
            print(f"[{i}/{len(chapters_data)}] 正在处理: {title}")
            
            if not content:
                summary = "[无内容]"
            else:
                summary = self.summarize_chapter(title, content)
            
            results.append({
                'chapter_num': i,
                'title': title,
                'summary': summary,
                'original_length': len(content)
            })
            
            print(f"完成概括 (原文{len(content)}字 -> 概括{len(summary)}字)")
        
        return results
    
    def create_final_summary(self, chapter_summaries: List[Dict]) -> str:
        """创建最终的完整概括"""
        print("正在生成最终概括...")
        
        # 构建完整的故事概括
        story_parts = []
        
        for chapter in chapter_summaries:
            if not chapter['summary'].startswith('['):  # 排除错误信息
                story_parts.append(chapter['summary'])
        
        # 合并为连贯的故事
        final_story = " ".join(story_parts)
        
        # 添加标题和统计信息
        total_chapters = len(chapter_summaries)
        total_original_length = sum(ch['original_length'] for ch in chapter_summaries)
        
        final_text = f"""《当家主母，我在现代整治顶级豪门》前{total_chapters}章概括

【故事梗概】
{final_story}

【统计信息】
- 总章节数：{total_chapters}章
- 原文总字数：{total_original_length:,}字
- 概括总字数：{len(final_story):,}字
- 压缩比：{len(final_story)/total_original_length*100:.1f}%"""
        
        return final_text
    
    def save_results(self, chapter_summaries: List[Dict], final_summary: str, output_file: str = "novel_summary.txt"):
        """保存结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入详细的章节概括
            f.write("=" * 60 + "\n")
            f.write("章节详细概括\n")
            f.write("=" * 60 + "\n\n")
            
            for chapter in chapter_summaries:
                f.write(f"【{chapter['title']}】\n")
                f.write(f"原文长度：{chapter['original_length']}字\n")
                f.write(f"概括：{chapter['summary']}\n")
                f.write("-" * 40 + "\n\n")
            
            # 写入最终概括
            f.write("\n" + "=" * 60 + "\n")
            f.write("完整故事概括\n")
            f.write("=" * 60 + "\n\n")
            f.write(final_summary)
        
        print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    # 可以在这里修改模型路径
    # 如果您有Qwen3-8B的访问权限，请修改下面的路径
    model_options = [
        "Qwen/Qwen2.5-3B-Instruct",  # 默认选项
        "Qwen/Qwen2.5-7B-Instruct",  # 更大的模型
        # "/path/to/your/qwen3-8b",   # 本地Qwen3-8B路径
    ]
    
    # 初始化概括器
    summarizer = Qwen3ChapterSummarizer(model_options[0])
    
    # 加载和解析数据
    chapters_data = summarizer.load_and_parse_csv("test.csv")
    if not chapters_data:
        print("❌ 数据加载失败，程序退出")
        return
    
    # 处理所有章节
    chapter_summaries = summarizer.process_all_chapters(chapters_data)
    
    # 生成最终概括
    final_summary = summarizer.create_final_summary(chapter_summaries)
    
    # 保存结果
    summarizer.save_results(chapter_summaries, final_summary)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📖 最终概括结果")
    print("=" * 60)
    print(final_summary)
    print("\n✅ 处理完成！")

if __name__ == "__main__":
    main()