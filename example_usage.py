#!/usr/bin/env python3
"""
Example demonstrating the improved data structure for fine-tuning
"""

import json

def show_data_structure():
    """Show the improved training data structure"""
    
    print("Improved Training Data Structure")
    print("=" * 50)
    
    # Example input data structure
    example_chapters = [
        {
            "title": "第01章 解开你的腰带",
            "content": "01 解开你的腰带\n"哎呀你别催了，这暴雨天路上堵得厉害，我也着急呀！"赵薇坐在副驾驶座上，一边系着安全带一边埋怨着开车的男人。"
        },
        {
            "title": "第02章 跪下",
            "content": "02 跪下\n林浩看着眼前的女人，心中五味杂陈。三年前的那个晚上改变了一切..."
        }
    ]
    
    example_revise_asr = "🔥震撼来袭！一场暴雨夜的邂逅，开启了命运的齿轮！赵薇与神秘男人的纠葛究竟会走向何方？三年前的秘密即将揭开，林浩的真实身份让人震惊！爱恨情仇交织，悬念迭起，每一章都让你欲罢不能！📖✨"
    
    print("📋 Raw Data Format:")
    print("-" * 30)
    print("chapters_text (JSON string):")
    print(json.dumps(example_chapters, ensure_ascii=False, indent=2))
    print("\nrevise_asr (target output):")
    print(example_revise_asr)
    
    print("\n🔄 After Preprocessing:")
    print("-" * 30)
    
    # Format chapters as readable text
    formatted_chapters = ""
    for chapter in example_chapters:
        title = chapter.get('title', '')
        content = chapter.get('content', '')
        formatted_chapters += f"章节标题: {title}\n章节内容: {content}\n\n"
    formatted_chapters = formatted_chapters.strip()
    
    # Create training example
    training_example = {
        "instruction": "请根据以下章节内容生成优秀的广告文案",
        "input": formatted_chapters,
        "output": example_revise_asr
    }
    
    print("Training Example Structure:")
    for key, value in training_example.items():
        print(f"\n{key.upper()}:")
        print(f"Length: {len(value)} characters")
        if len(value) > 200:
            print(f"Content: {value[:200]}...")
        else:
            print(f"Content: {value}")
    
    print("\n🎯 What the Model Learns:")
    print("-" * 30)
    print("INPUT: chapters_text (formatted chapter content)")
    print("↓")
    print("PROCESSING: Fine-tuned Qwen model")
    print("↓")
    print("OUTPUT: revise_asr (excellent advertising copy)")
    
    print("\n✅ Benefits of This Structure:")
    print("-" * 30)
    print("1. Clear separation: instruction vs. actual content")
    print("2. Direct mapping: chapters_text → revise_asr")
    print("3. Focused learning: model learns to transform content into ads")
    print("4. Consistent format: same instruction, different inputs")
    print("5. Better generalization: can work with any chapter content")
    
    return training_example

def show_conversation_format():
    """Show how the data is formatted for Qwen training"""
    
    print("\n🗣️ Conversation Format for Training:")
    print("=" * 50)
    
    example = show_data_structure()
    
    # Format as Qwen conversation
    user_message = f"{example['instruction']}\n\n{example['input']}"
    conversation = f"""<|im_start|>system
你是一个专业的广告文案创作助手，擅长根据详细的内容创作吸引人的广告文案。<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
    
    print("Final Conversation Format:")
    print("-" * 30)
    print(conversation)
    
    print(f"\nTotal conversation length: {len(conversation)} characters")

if __name__ == "__main__":
    show_data_structure()
    show_conversation_format()