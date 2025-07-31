import csv
import json
import ast

def simple_test():
    """
    简化的数据格式测试
    """
    print("🔍 简化测试：检查CSV文件格式")
    print("=" * 50)
    
    try:
        # 直接读取CSV文件
        with open('test.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # 读取头部
            data_row = next(csv_reader)  # 读取第一行数据
            
        print(f"✅ CSV文件读取成功")
        print(f"📊 列数: {len(headers)}")
        
        # 查找chapters_text列的索引
        try:
            chapters_text_index = headers.index('chapters_text')
            print(f"✅ 找到 'chapters_text' 列，位置: {chapters_text_index}")
        except ValueError:
            print(f"❌ 没有找到 'chapters_text' 列")
            print(f"📊 可用列名: {headers}")
            return
        
        # 获取chapters_text数据
        chapters_text = data_row[chapters_text_index]
        print(f"\n📝 chapters_text 数据信息:")
        print(f"   数据类型: {type(chapters_text)}")
        print(f"   数据长度: {len(chapters_text)} 字符")
        
        # 显示前500个字符
        preview = chapters_text[:500] + "..." if len(chapters_text) > 500 else chapters_text
        print(f"   数据预览: {preview}")
        
        # 尝试解析
        print(f"\n🔄 尝试解析章节数据...")
        
        # 方法1: JSON解析
        try:
            result = json.loads(chapters_text)
            if isinstance(result, list) and len(result) > 0:
                print(f"✅ JSON解析成功！")
                print(f"📚 章节数量: {len(result)}")
                
                # 显示前3个章节
                for i, chapter in enumerate(result[:3]):
                    if isinstance(chapter, dict):
                        title = chapter.get('title', f'第{i+1}章')
                        content = chapter.get('content', '')
                        print(f"   📖 第{i+1}章: {title} ({len(content)} 字)")
                
                return True
            else:
                print(f"❌ JSON解析结果不是有效的章节列表")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {str(e)[:100]}")
        
        # 方法2: AST解析
        try:
            result = ast.literal_eval(chapters_text)
            if isinstance(result, list) and len(result) > 0:
                print(f"✅ AST解析成功！")
                print(f"📚 章节数量: {len(result)}")
                
                # 显示前3个章节
                for i, chapter in enumerate(result[:3]):
                    if isinstance(chapter, dict):
                        title = chapter.get('title', f'第{i+1}章')
                        content = chapter.get('content', '')
                        print(f"   📖 第{i+1}章: {title} ({len(content)} 字)")
                
                return True
            else:
                print(f"❌ AST解析结果不是有效的章节列表")
                
        except (ValueError, SyntaxError) as e:
            print(f"❌ AST解析失败: {str(e)[:100]}")
        
        print(f"❌ 所有解析方法都失败")
        return False
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 test.csv 文件")
        return False
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    
    if success:
        print(f"\n✅ 数据格式验证成功！")
        print(f"🚀 可以安装依赖并运行主程序:")
        print(f"   pip install -r requirements.txt")
        print(f"   python chapter_summarizer_fixed.py")
    else:
        print(f"\n❌ 数据格式验证失败")
        print(f"🔧 请检查 test.csv 文件中的 chapters_text 列格式")