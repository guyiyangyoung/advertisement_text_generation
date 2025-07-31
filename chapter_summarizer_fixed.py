import pandas as pd
import json
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List, Dict, Optional

class ChapterSummarizerFixed:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        初始化章节概括器
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 正在加载模型: {model_path}")
        print(f"💻 使用设备: {self.device}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
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
            
            print("✅ 模型加载完成!")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def parse_chapters_data(self, chapters_text) -> Optional[List[Dict]]:
        """
        解析chapters_text数据，支持多种格式
        """
        print("📖 正在解析章节数据...")
        
        # 如果已经是list，直接返回
        if isinstance(chapters_text, list):
            print(f"✅ 数据已是list格式，包含 {len(chapters_text)} 个章节")
            return chapters_text
        
        # 如果是字符串，尝试解析
        if isinstance(chapters_text, str):
            # 尝试多种解析方法
            parsing_methods = [
                ("JSON解析", lambda x: json.loads(x)),
                ("AST解析", lambda x: ast.literal_eval(x)),
                ("修复后JSON解析", self._fix_and_parse_json),
            ]
            
            for method_name, parse_func in parsing_methods:
                try:
                    print(f"🔄 尝试使用 {method_name}...")
                    result = parse_func(chapters_text)
                    if isinstance(result, list) and len(result) > 0:
                        print(f"✅ {method_name} 成功，解析出 {len(result)} 个章节")
                        return result
                except Exception as e:
                    print(f"❌ {method_name} 失败: {str(e)[:100]}")
                    continue
        
        print("❌ 所有解析方法都失败了")
        return None
    
    def _fix_and_parse_json(self, text: str) -> List[Dict]:
        """
        尝试修复并解析JSON格式的文本
        """
        # 清理文本
        text = text.strip()
        
        # 如果不是以[开头，寻找第一个[
        if not text.startswith('['):
            start_idx = text.find('[')
            if start_idx != -1:
                text = text[start_idx:]
        
        # 如果不是以]结尾，寻找最后一个]
        if not text.endswith(']'):
            end_idx = text.rfind(']')
            if end_idx != -1:
                text = text[:end_idx + 1]
        
        # 尝试解析
        return json.loads(text)
    
    def load_and_parse_csv(self, csv_file: str) -> Optional[List[Dict]]:
        """
        加载CSV文件并解析chapters_text列
        """
        print(f"📁 正在读取文件: {csv_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            print(f"📊 CSV文件列名: {list(df.columns)}")
            print(f"📊 数据行数: {len(df)}")
            
            if 'chapters_text' not in df.columns:
                print("❌ 错误: CSV文件中没有找到 'chapters_text' 列")
                return None
            
            # 获取chapters_text列的数据
            chapters_text = df['chapters_text'].iloc[0]
            print(f"📝 chapters_text 数据类型: {type(chapters_text)}")
            print(f"📝 数据长度: {len(str(chapters_text))}")
            
            # 显示数据的前200个字符
            preview = str(chapters_text)[:200] + "..." if len(str(chapters_text)) > 200 else str(chapters_text)
            print(f"📝 数据预览: {preview}")
            
            # 解析章节数据
            chapters_data = self.parse_chapters_data(chapters_text)
            
            if chapters_data:
                # 显示前几个章节的信息
                print(f"\n📚 成功解析出 {len(chapters_data)} 个章节:")
                for i, chapter in enumerate(chapters_data[:3]):
                    title = chapter.get('title', f'第{i+1}章')
                    content_length = len(chapter.get('content', ''))
                    print(f"  📖 章节 {i+1}: {title} (内容长度: {content_length} 字)")
                
                if len(chapters_data) > 3:
                    print(f"  ... 还有 {len(chapters_data) - 3} 个章节")
            
            return chapters_data
            
        except Exception as e:
            print(f"❌ 文件读取错误: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        清理文本内容
        """
        if not text:
            return ""
        
        # 移除章节编号（如果在内容开头）
        text = re.sub(r'^\d+\s+', '', text.strip())
        
        # 移除换行符和多余空格
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # 移除引号和对话标记
        text = re.sub(r'^["""]', '', text)
        text = re.sub(r'["""]$', '', text)
        
        # 保留中文、英文、数字和常用标点
        text = re.sub(r'[^\u4e00-\u9fff\u0030-\u0039\u0041-\u005a\u0061-\u007a\s，。！？；：""''（）《》【】、]', '', text)
        
        return text.strip()
    
    def create_summary_prompt(self, title: str, content: str) -> str:
        """
        创建概括提示词
        """
        return f"""请将以下小说章节概括为约200字的精炼文字。

要求：
1. 提取关键情节和人物行为
2. 保持故事的连贯性和逻辑性
3. 语言简洁流畅，突出重点
4. 字数控制在180-220字之间

章节标题：{title}

章节内容：
{content}

请开始概括："""
    
    def summarize_chapter(self, title: str, content: str, max_input_length: int = 1800) -> str:
        """
        使用模型对单个章节进行概括
        """
        # 清理内容
        cleaned_content = self.clean_text(content)
        
        # 控制输入长度
        if len(cleaned_content) > max_input_length:
            cleaned_content = cleaned_content[:max_input_length] + "..."
        
        # 如果内容太短，直接返回
        if len(cleaned_content) < 20:
            return "[内容过短，无法概括]"
        
        # 创建提示词
        prompt = self.create_summary_prompt(title, cleaned_content)
        
        try:
            # 构建对话
            messages = [
                {
                    "role": "system", 
                    "content": "你是一个专业的文学作品概括专家，擅长提取故事的核心情节和关键信息，用简洁的语言进行精准概括。"
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            
            # 移动到设备
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # 生成概括
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=280,
                    temperature=0.3,
                    top_p=0.85,
                    do_sample=True,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # 清理输出
            summary = generated_text.strip()
            
            # 移除可能的重复内容或无关文本
            if "概括：" in summary:
                summary = summary.split("概括：")[-1].strip()
            
            return summary if summary else "[生成失败]"
            
        except Exception as e:
            print(f"❌ 生成概括时出错: {e}")
            return f"[生成失败: {str(e)}]"
    
    def process_all_chapters(self, chapters_data: List[Dict]) -> List[Dict]:
        """
        处理所有章节
        """
        results = []
        total_chapters = len(chapters_data)
        
        print(f"\n🔄 开始处理 {total_chapters} 个章节...")
        print("=" * 60)
        
        for i, chapter in enumerate(chapters_data, 1):
            title = chapter.get('title', f'第{i}章')
            content = chapter.get('content', '')
            
            print(f"📖 [{i:2d}/{total_chapters}] 正在处理: {title}")
            
            if not content.strip():
                summary = "[章节无内容]"
                print(f"⚠️  警告: 章节内容为空")
            else:
                print(f"📝 原文长度: {len(content)} 字")
                summary = self.summarize_chapter(title, content)
                print(f"✅ 概括完成: {len(summary)} 字")
            
            results.append({
                'chapter_num': i,
                'title': title,
                'summary': summary,
                'original_length': len(content),
                'summary_length': len(summary)
            })
            
            print("-" * 40)
        
        return results
    
    def create_integrated_summary(self, chapter_summaries: List[Dict]) -> str:
        """
        创建整合的故事概括
        """
        print("\n🔄 正在生成整合概括...")
        
        # 收集有效的概括
        valid_summaries = []
        for chapter in chapter_summaries:
            summary = chapter['summary']
            if not summary.startswith('[') and len(summary.strip()) > 10:
                valid_summaries.append(summary)
        
        # 统计信息
        total_chapters = len(chapter_summaries)
        total_original_length = sum(ch['original_length'] for ch in chapter_summaries)
        total_summary_length = sum(ch['summary_length'] for ch in chapter_summaries)
        
        # 构建最终概括
        integrated_story = " ".join(valid_summaries)
        
        final_text = f"""《当家主母，我在现代整治顶级豪门》前{total_chapters}章概括

【故事梗概】
{integrated_story}

【统计信息】
• 总章节数：{total_chapters} 章
• 原文总字数：{total_original_length:,} 字
• 概括总字数：{total_summary_length:,} 字
• 有效概括章节：{len(valid_summaries)} 章
• 内容压缩比：{(total_summary_length/total_original_length*100):.1f}%

【处理完成时间】
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        return final_text
    
    def save_results(self, chapter_summaries: List[Dict], final_summary: str, output_file: str = "novel_summary_result.txt"):
        """
        保存处理结果
        """
        print(f"\n💾 正在保存结果到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入标题
            f.write("=" * 80 + "\n")
            f.write("小说章节概括结果\n")
            f.write("=" * 80 + "\n\n")
            
            # 写入章节详细概括
            f.write("【章节详细概括】\n")
            f.write("-" * 60 + "\n\n")
            
            for chapter in chapter_summaries:
                f.write(f"章节编号：{chapter['chapter_num']}\n")
                f.write(f"章节标题：{chapter['title']}\n")
                f.write(f"原文长度：{chapter['original_length']} 字\n")
                f.write(f"概括长度：{chapter['summary_length']} 字\n")
                f.write(f"概括内容：{chapter['summary']}\n")
                f.write("\n" + "—" * 50 + "\n\n")
            
            # 写入整合概括
            f.write("\n" + "=" * 80 + "\n")
            f.write("【整合故事概括】\n")
            f.write("=" * 80 + "\n\n")
            f.write(final_summary)
        
        print(f"✅ 结果已成功保存到: {output_file}")

def main():
    """
    主函数
    """
    print("🎯 小说章节概括工具")
    print("=" * 50)
    
    try:
        # 初始化概括器
        summarizer = ChapterSummarizerFixed()
        
        # 加载和解析数据
        chapters_data = summarizer.load_and_parse_csv("test.csv")
        if not chapters_data:
            print("❌ 数据加载失败，程序退出")
            return
        
        # 处理所有章节
        chapter_summaries = summarizer.process_all_chapters(chapters_data)
        
        # 生成整合概括
        final_summary = summarizer.create_integrated_summary(chapter_summaries)
        
        # 保存结果
        summarizer.save_results(chapter_summaries, final_summary)
        
        # 显示最终结果
        print("\n" + "=" * 80)
        print("🎉 处理完成！最终概括结果：")
        print("=" * 80)
        print(final_summary)
        
        print("\n✅ 全部处理完成！请查看输出文件 'novel_summary_result.txt'")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()