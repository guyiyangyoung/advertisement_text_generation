import pandas as pd
import json
import ast

def test_data_parsing():
    """
    测试数据解析功能
    """
    print("🔍 测试CSV数据解析功能")
    print("=" * 50)
    
    try:
        # 读取CSV文件
        df = pd.read_csv("test.csv")
        print(f"✅ CSV文件读取成功")
        print(f"📊 数据行数: {len(df)}")
        print(f"📊 列名: {list(df.columns)}")
        
        if 'chapters_text' not in df.columns:
            print("❌ 错误: 没有找到 'chapters_text' 列")
            return
        
        # 获取chapters_text数据
        chapters_text = df['chapters_text'].iloc[0]
        print(f"\n📝 chapters_text 数据信息:")
        print(f"   数据类型: {type(chapters_text)}")
        print(f"   数据长度: {len(str(chapters_text))} 字符")
        
        # 显示数据预览
        preview = str(chapters_text)[:300] + "..." if len(str(chapters_text)) > 300 else str(chapters_text)
        print(f"   数据预览: {preview}")
        
        # 尝试不同的解析方法
        print(f"\n🔄 尝试解析数据...")
        
        # 方法1: 直接JSON解析
        try:
            result1 = json.loads(chapters_text)
            print(f"✅ JSON解析成功！解析出 {len(result1)} 个章节")
            
            # 显示前3个章节信息
            for i, chapter in enumerate(result1[:3]):
                title = chapter.get('title', f'第{i+1}章')
                content_len = len(chapter.get('content', ''))
                print(f"   📖 章节 {i+1}: {title} (内容: {content_len} 字)")
            
            return result1
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {str(e)[:100]}")
        
        # 方法2: AST解析（如果数据是Python字面量格式）
        try:
            result2 = ast.literal_eval(chapters_text)
            print(f"✅ AST解析成功！解析出 {len(result2)} 个章节")
            
            # 显示前3个章节信息
            for i, chapter in enumerate(result2[:3]):
                title = chapter.get('title', f'第{i+1}章')
                content_len = len(chapter.get('content', ''))
                print(f"   📖 章节 {i+1}: {title} (内容: {content_len} 字)")
            
            return result2
            
        except (ValueError, SyntaxError) as e:
            print(f"❌ AST解析失败: {str(e)[:100]}")
        
        # 方法3: 尝试修复JSON格式
        try:
            # 查找方括号
            text = str(chapters_text).strip()
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_text = text[start_idx:end_idx+1]
                result3 = json.loads(json_text)
                print(f"✅ 修复后JSON解析成功！解析出 {len(result3)} 个章节")
                
                # 显示前3个章节信息
                for i, chapter in enumerate(result3[:3]):
                    title = chapter.get('title', f'第{i+1}章')
                    content_len = len(chapter.get('content', ''))
                    print(f"   📖 章节 {i+1}: {title} (内容: {content_len} 字)")
                
                return result3
            else:
                print(f"❌ 无法找到有效的JSON结构")
                
        except json.JSONDecodeError as e:
            print(f"❌ 修复后JSON解析仍然失败: {str(e)[:100]}")
        
        print("❌ 所有解析方法都失败了")
        return None
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return None

def analyze_chapter_content(chapters_data):
    """
    分析章节内容
    """
    if not chapters_data:
        return
    
    print(f"\n📊 章节内容分析:")
    print("=" * 50)
    
    total_content_length = 0
    
    for i, chapter in enumerate(chapters_data, 1):
        title = chapter.get('title', f'第{i}章')
        content = chapter.get('content', '')
        content_length = len(content)
        total_content_length += content_length
        
        print(f"第{i:2d}章: {title}")
        print(f"      内容长度: {content_length} 字")
        
        # 显示内容开头
        if content:
            preview = content[:100].replace('\n', ' ')
            print(f"      内容预览: {preview}...")
        else:
            print(f"      ⚠️ 内容为空")
        
        print()
    
    print(f"📈 统计信息:")
    print(f"   总章节数: {len(chapters_data)}")
    print(f"   总字数: {total_content_length:,}")
    print(f"   平均字数: {total_content_length // len(chapters_data):,}")

if __name__ == "__main__":
    # 执行测试
    chapters_data = test_data_parsing()
    
    if chapters_data:
        analyze_chapter_content(chapters_data)
        print("\n✅ 数据解析测试完成！可以运行主程序了。")
    else:
        print("\n❌ 数据解析测试失败，请检查数据格式。")