import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class ChapterSummarizer:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        """
        初始化章节概括器
        注意：由于Qwen3-8B可能需要特殊权限，这里使用Qwen2.5-3B-Instruct作为替代
        """
        print("正在加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("模型加载完成!")
    
    def load_data(self, csv_file):
        """
        加载CSV文件并解析chapters_text列
        """
        print(f"正在读取文件: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 获取chapters_text列的数据
        chapters_text = df['chapters_text'].iloc[0]  # 假设只有一行数据
        
        # 解析JSON数据
        try:
            chapters_data = json.loads(chapters_text)
            print(f"成功解析出 {len(chapters_data)} 个章节")
            return chapters_data
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    
    def clean_text(self, text):
        """
        清理文本，去除多余的空白字符和特殊字符
        """
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊的Unicode字符
        text = re.sub(r'[^\u4e00-\u9fff\u0030-\u0039\u0041-\u005a\u0061-\u007a\s，。！？；：""''（）《》【】]', '', text)
        return text.strip()
    
    def summarize_chapter(self, chapter_title, chapter_content):
        """
        使用Qwen模型对单个章节进行概括
        """
        # 清理章节内容
        cleaned_content = self.clean_text(chapter_content)
        
        # 如果内容太长，只取前2000个字符
        if len(cleaned_content) > 2000:
            cleaned_content = cleaned_content[:2000] + "..."
        
        # 构建提示词
        prompt = f"""请将以下章节内容概括为200字左右的文字：

章节标题：{chapter_title}

章节内容：
{cleaned_content}

请用简洁明了的语言概括这一章的主要情节和关键信息，字数控制在200字左右："""

        # 构建对话格式
        messages = [
            {"role": "system", "content": "你是一个专业的文本概括助手，擅长将长篇内容概括为简洁明了的摘要。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            model_inputs = model_inputs.to(self.model.device)
        
        # 生成概括
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def process_all_chapters(self, chapters_data):
        """
        处理所有章节并生成概括
        """
        chapter_summaries = []
        
        print("开始处理章节...")
        for i, chapter in enumerate(chapters_data, 1):
            title = chapter.get('title', f'第{i}章')
            content = chapter.get('content', '')
            
            print(f"正在处理: {title}")
            
            try:
                summary = self.summarize_chapter(title, content)
                chapter_summaries.append({
                    'chapter_num': i,
                    'title': title,
                    'summary': summary
                })
                print(f"完成: {title}")
            except Exception as e:
                print(f"处理章节 {title} 时出错: {e}")
                chapter_summaries.append({
                    'chapter_num': i,
                    'title': title,
                    'summary': f"[处理失败] {str(e)}"
                })
        
        return chapter_summaries
    
    def generate_final_summary(self, chapter_summaries):
        """
        将所有章节概括整合为一段完整的文字
        """
        print("正在生成最终概括...")
        
        # 收集所有章节概括
        all_summaries = []
        for chapter in chapter_summaries:
            summary_text = f"{chapter['title']}：{chapter['summary']}"
            all_summaries.append(summary_text)
        
        # 整合为一段文字
        final_text = "《当家主母，我在现代整治顶级豪门》前20章概括：\n\n"
        final_text += " ".join(all_summaries)
        
        return final_text
    
    def save_results(self, chapter_summaries, final_summary, output_file="chapter_summaries.txt"):
        """
        保存结果到文件
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("章节详细概括\n")
            f.write("=" * 50 + "\n\n")
            
            for chapter in chapter_summaries:
                f.write(f"【{chapter['title']}】\n")
                f.write(f"{chapter['summary']}\n\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("整体概括\n")
            f.write("=" * 50 + "\n\n")
            f.write(final_summary)
        
        print(f"结果已保存到: {output_file}")

def main():
    """
    主函数
    """
    # 初始化概括器
    summarizer = ChapterSummarizer()
    
    # 加载数据
    chapters_data = summarizer.load_data("test.csv")
    if chapters_data is None:
        print("数据加载失败，程序退出")
        return
    
    # 处理所有章节
    chapter_summaries = summarizer.process_all_chapters(chapters_data)
    
    # 生成最终概括
    final_summary = summarizer.generate_final_summary(chapter_summaries)
    
    # 保存结果
    summarizer.save_results(chapter_summaries, final_summary)
    
    # 打印最终概括
    print("\n" + "=" * 60)
    print("最终概括结果：")
    print("=" * 60)
    print(final_summary)

if __name__ == "__main__":
    main()