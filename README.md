# 章节概括工具

这个工具用于解析CSV文件中的小说章节数据，并使用Qwen模型对每个章节进行概括。

## 文件说明

- `chapter_summarizer.py` - 基础版本的章节概括脚本
- `chapter_summarizer_qwen3.py` - 专门针对Qwen3-8B优化的版本（推荐使用）
- `requirements.txt` - 依赖包列表
- `test.csv` - 输入数据文件（包含chapters_text列）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法1：使用基础版本
```bash
python chapter_summarizer.py
```

### 方法2：使用Qwen3优化版本（推荐）
```bash
python chapter_summarizer_qwen3.py
```

## 配置模型

如果您有Qwen3-8B模型的访问权限，可以修改脚本中的模型路径：

在 `chapter_summarizer_qwen3.py` 中找到以下部分：
```python
model_options = [
    "Qwen/Qwen2.5-3B-Instruct",  # 默认选项
    "Qwen/Qwen2.5-7B-Instruct",  # 更大的模型
    # "/path/to/your/qwen3-8b",   # 本地Qwen3-8B路径
]
```

将您的Qwen3-8B模型路径添加到列表中，然后修改初始化行：
```python
summarizer = Qwen3ChapterSummarizer("/path/to/your/qwen3-8b")
```

## 输入数据格式

CSV文件需要包含 `chapters_text` 列，该列存储JSON格式的章节数据：

```json
[
  {
    "title": "第01章 章节标题",
    "content": "章节内容..."
  },
  {
    "title": "第02章 章节标题",
    "content": "章节内容..."
  }
]
```

## 输出结果

脚本会生成以下文件：
- `novel_summary.txt` - 包含详细的章节概括和整体故事概括

输出包含：
1. 每个章节的详细概括（约200字）
2. 完整的故事概括（所有章节概括的整合）
3. 统计信息（字数、压缩比等）

## 系统要求

- Python 3.8+
- 8GB+ RAM（推荐16GB+）
- CUDA支持的GPU（可选，会显著提升速度）

## 注意事项

1. 首次运行会自动下载模型文件，需要较长时间和稳定的网络连接
2. 如果没有GPU，处理速度会较慢
3. 如果遇到模型加载问题，脚本会自动切换到备用模型
4. 生成的概括可能需要人工审校以确保质量

## 故障排除

### 模型下载失败
确保网络连接稳定，或者手动下载模型到本地目录。

### 内存不足
尝试使用较小的模型，或者减少batch_size。

### JSON解析错误
检查CSV文件中的chapters_text列格式是否正确。

## 自定义配置

您可以在脚本中调整以下参数：
- `max_new_tokens`: 生成文本的最大长度
- `temperature`: 生成随机性（0-1）
- `max_length`: 输入文本的最大长度
- 概括字数要求（默认200字左右）