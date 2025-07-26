import pandas as pd
import sys 
import os
from openai import OpenAI
import json
import re
import time
import torch
from multiprocessing import Process
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 设置环境变量避免tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 模型路径
model_name = "/mnt/bn/ug-diffusion-yg-nas/guyiyang/Qwen3-8B"


def worker(gpu_id, row_idxs, df_path, output_file_path, batch_size=8):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Tokenizer加载成功")
    except Exception as e:
        print(f"Tokenizer加载失败: {e}")
        exit(1)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).eval()
        model = model.to(device)
        print(f"模型已成功加载到 GPU {gpu_id}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)

    def get_asr_result_qwen_batch(batch_messages):
        """
        批量处理消息
        batch_messages: list of messages, 每个元素是一个对话消息列表
        """
        outputs = None
        try:
            # 为每个对话应用聊天模板
            batch_inputs = []
            for messages in batch_messages:
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt",
                    padding=False,  # 先不padding，后面统一处理
                    truncation=True,
                    max_length=16384
                )
                batch_inputs.append(inputs.squeeze(0))  # 移除batch维度
            
            # 统一padding到最大长度
            max_length = max(len(inp) for inp in batch_inputs)
            padded_inputs = []
            attention_masks = []
            
            for inp in batch_inputs:
                # 左padding
                pad_length = max_length - len(inp)
                if pad_length > 0:
                    padded_inp = torch.cat([torch.full((pad_length,), tokenizer.pad_token_id), inp])
                    attention_mask = torch.cat([torch.zeros(pad_length), torch.ones(len(inp))])
                else:
                    padded_inp = inp
                    attention_mask = torch.ones(len(inp))
                
                padded_inputs.append(padded_inp)
                attention_masks.append(attention_mask)
            
            # 转换为tensor
            inputs = torch.stack(padded_inputs).to(device)
            attention_mask = torch.stack(attention_masks).long().to(device)
            
            print(f"批处理输入张量形状: {inputs.shape}")
            print(f"批处理attention mask形状: {attention_mask.shape}")
            print(f"输入张量设备: {inputs.device}")
            print(f"Attention mask设备: {attention_mask.device}")
        except Exception as e:
            print(f"批处理输入处理失败: {e}")
            return [None] * len(batch_messages)

        generate_config = {
            "max_new_tokens": 16384,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.35,
            "attention_mask": attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    **generate_config
                )
            
            # 解码每个输出
            response_texts = []
            for i, output in enumerate(outputs):
                response_text = tokenizer.decode(output, skip_special_tokens=True)
                response_texts.append(response_text)
            
            return response_texts
        except Exception as e:
            print(f"批处理生成回答失败: {e}")
            print("\n当前输入设备:", inputs.device)
            print("首个参数设备:", next(model.parameters()).device)
            return [None] * len(batch_messages)
        
    def extract_json_after_output(response_text):
        # 匹配 #输出 后的首个 json 块（支持跨行、忽略前空格）
        match = re.search(r'#输出\s*([\s\S]*?{[\s\S]*})', response_text)
        if match:
            # 去掉前面的无关字符，仅保留第一个完整 { ... }
            possible_json = match.group(1).strip()
            # 查找第一个和最后一个大括号范围
            start = possible_json.find('{')
            end = possible_json.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = possible_json[start:end+1]
                try:
                    return json.loads(json_str)
                except Exception as e:
                    print("JSON解析失败:", e)
                    print("内容为:", json_str)
                    return None
            else:
                print("没有找到有效的大括号包裹内容。")
                return None
        else:
            print("没有找到 #输出 后的JSON。")
            return None
    
    def revise_asr_batch(asr_list, ocr_list):
        """
        批量处理ASR和OCR文本
        """
        demo_system_text = f'''
        # 角色:
        你是一名专业的文案优化师，擅长将长篇ASR文本和OCR文本，修改为专业脚本文案。

        ## 目标:
        - 去掉末尾不完整的句子。
        - 去掉末尾营销号召的文字，如果有的话。比如，"点击下方链接，免费获取全文！"，"来番茄小说，发现更多精彩！"，"点击下方链接，即可观看全文！"，"点击视频下方，继续阅读全文！"
        - 根据OCR修正ASR发音相同或相似的错别字。ASR文本存在着发音相同或者相似的错别字，找出所有用词不当或者语义不明的词，结合OCR和上下文理解修改这些错别字。尤其注意人名、地名、物名等。
        - 修正涉政涉黄的词语，如政治人物、政治事件、党派描述、国家象征、国旗国徽、军装等。
        - 使用吸睛的话术把文案改得更有吸引力。明确指出谁在说话，可以多使用转折词（例如：突然、一下子、回过头来等），上下对话之间要有问答关联（比如：是啊，对啊等词汇），快速让人明白存在的冲突，吸引人注意。

        ## 约束:
        - 文案尽量与ASR文案一致，字数也尽量与ASR文案接近
        - ASR用词不当或者语义不明的地方，尽量使用对应的OCR文字来修正
        - 只允许修改ASR错别字，使用话术增加文案吸引力，语义包括细节不能有变化，修改比例控制在20%以内。
        - 整体文案要有吸引力
        - 整体文案剧情紧凑且逻辑通顺。
        - 不能涉及政治：不能出现政治立场、政治人物、政治事件、党派描述、国家象征、国旗国徽、军装等，如果有用虚指代替
        - 不能出现跟中国政府机关有关的地点名称，比如民政局、纪委、检察院、税务所、公安局、派出所等，出现这种机构要用相应的普通地点或者虚指代替，比如大厅、屋子、那里等
        
        ## 内容审查与输出：  
        - 审查所有内容是否符合限制要求，如禁止政治、色情、暴力等。  

        ## 输出格式：
        - JSON字典格式，包含以下字段：
        - **文案**：修改后的文案
        - **修改说明**：一些主要修改说明

        
        ## 示例:
        示例一：  
        输入：
        ASR: 老爷子的话语让汪瓶有种心惊肉跳的感觉，纪委的事情只有自己一个人知道，他怎么突然问起这个了 
        OCR: 老爷子的话语让王平有种心惊肉跳的感觉，纪委的事情只有自己一个人知道，他怎么突然问起这个了 
        输出：
        {{
            "文案": "老爷子的话语让王平有种心惊肉跳的感觉，那里的事情只有自己一个人知道，他怎么突然问起这个了",
            "修改说明":""王平"等人名以OCR为准，"纪委"是涉政敏感词，修改成代词"那里"。"
            }}

        示例二：  
        输入：
        ASR: 杨紫不在家，你老是跟我说，你时不时出去吃烧烤啦
        OCR: 养子不在家，你老实跟我说，你时不时出去吃烧烤啦
        输出：     
        {{
            "文案": "养子不在家，你老实给我说，你是不是出去吃烧烤啦",
            "修改说明":"'养子'以OCR为准，结合OCR和上下文，"老是"是asr错别字，应该是发音相似的"老实"，"不时时"是asr错别字，应该是发音相似的"是不是"。"
            }}

        示例三：
        输入：
        ASR：军医院总院长汪萍死死抱住一位佩戴着玉佩的年轻男人，...凝视着年轻人，越看越觉得他与自己的弟弟相像，而且当初弟弟下去劳动回来，隆沛就不见了...小伙子就叫王浩，他身上还有您给弟弟的龙佩 
        OCR：军医院总院长王平死死抱住一位佩戴着玉佩的年轻男人，...凝视着年轻人，越看越觉得他与自己的弟弟相像，而且当初弟弟下去劳动回来，隆沛就不见了...小伙子就叫王浩，他身上还有您给弟弟的龙佩 
        输出： 
        {{
            "文案": "军医院总院长王平死死抱住一位佩戴着玉佩的年轻男人，...凝视着年轻人，越看越觉得他与自己的弟弟相像，而且当初弟弟下去劳动回来，龙佩就不见了...小伙子就叫王浩，他身上还有您给弟弟的龙佩",
            "修改说明":""王平"等人名以OCR为准，结合上下文，"隆沛"是asr错别字，应该是发音相似的"龙佩"。"
            }}
        示例四：  
        输入：
        ASR：老爷子成因后说："你们都在那等我，我马上过来。" 
        OCR：老爷子沉吟后说："你们都在那等我，我马上过来。"
        输出：
        {{
            "文案": "老爷子稍微沉吟了一下，突然坚定地说道，"你们都在那里等我，我马上过来。"",
            "修改说明":"'沉吟'以OCR为准，加入"突然"等转折词，快速让人明白存在的冲突，吸引人注意。"
            }}  
        示例五：  
        输入：
        ASR：他看着他，深情的问到"他怎么样了" 
        OCR：她看着他，深情的问到，她怎么样了 
        输出：
        {{
            "文案": "她看着他，深情的问到"她怎么样了"",
            "修改说明":"'她'以OCR为准"
            }}  
        '''

        # 构建批量消息
        batch_messages = []
        for asr, ocr in zip(asr_list, ocr_list):
            content_text = f'''
        #输入
        ASR:{asr}
        OCR:{ocr}
        #输出
        '''
            messages = [
                {"role": "system", "content": demo_system_text},
                {"role": "user", "content": content_text},
            ]
            batch_messages.append(messages)

        # 批量生成结果
        batch_results = get_asr_result_qwen_batch(batch_messages)
        
        # 解析每个结果
        res_dicts = []
        for result in batch_results:
            if result is not None:
                res_dict = extract_json_after_output(result)
                res_dicts.append(res_dict)
            else:
                res_dicts.append(None)
        
        return res_dicts

    # 读取需要处理的全部行
    df = pd.read_csv(df_path)
    first_write = True
    
    # 按批次处理数据
    for i in range(0, len(row_idxs), batch_size):
        batch_indices = row_idxs[i:i+batch_size]
        print(f"处理批次: {batch_indices}")
        
        # 收集当前批次的数据
        batch_asr = []
        batch_ocr = []
        batch_categories = []
        
        for index in batch_indices:
            asr = df.loc[index, "raw_asr"]
            ocr = df.loc[index, "raw_ocr"]
            category_v2_name = df.loc[index, "category_v2_name"]
            
            batch_asr.append(asr)
            batch_ocr.append(ocr)
            batch_categories.append(category_v2_name)
        
        # 过滤掉"评书"类别的数据
        valid_indices = []
        valid_asr = []
        valid_ocr = []
        
        for j, (index, category) in enumerate(zip(batch_indices, batch_categories)):
            if "评书" in category:
                df.at[index, 'revise_asr'] = 'error1'
            else:
                valid_indices.append(index)
                valid_asr.append(batch_asr[j])
                valid_ocr.append(batch_ocr[j])
        
        # 批量处理有效数据
        if valid_indices:
            try:
                batch_results = revise_asr_batch(valid_asr, valid_ocr)
                print(f"批次结果: {len(batch_results)} 个结果")
                
                # 更新DataFrame
                for k, (index, result) in enumerate(zip(valid_indices, batch_results)):
                    if result and '文案' in result:
                        df.at[index, 'revise_asr'] = result['文案']
                    else:
                        df.at[index, 'revise_asr'] = ""
            except Exception as e:
                print(f"批次处理错误: {e}")
                # 如果批次处理失败，标记为错误
                for index in valid_indices:
                    df.at[index, 'revise_asr'] = "error2"
        
        # 流式写入当前批次的行
        batch_df = df.loc[batch_indices]
        if first_write:
            batch_df.to_csv(output_file_path, index=False)
            first_write = False
        else:
            batch_df.to_csv(output_file_path, mode='a', header=False, index=False)

def main():
    df_path = '/mnt/bn/ug-diffusion-yg-nas/guyiyang/data/all_generated_scripts_costlargerthan_1000.csv'
    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPUs')

    df = pd.read_csv(df_path)
    total_rows = len(df)

    # 将每一行编号分配给不同GPU
    row_indices = [[] for _ in range(num_gpus)]
    for i in range(total_rows):
        row_indices[i % num_gpus].append(i)

    # 多进程准备
    processes = []
    for gpu_id in range(num_gpus):
        output_file_path = f'/mnt/bn/ug-diffusion-yg-nas/guyiyang/data/revise_asr_parallel_{gpu_id}.csv'
        p = Process(target=worker, args=(gpu_id, row_indices[gpu_id], df_path, output_file_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并所有输出文件
    all_dfs = []
    for gpu_id in range(num_gpus):
        output_file_path = f'/mnt/bn/ug-diffusion-yg-nas/guyiyang/data/revise_asr_parallel_{gpu_id}.csv'
        if os.path.exists(output_file_path):
            df = pd.read_csv(output_file_path)
            all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_output_file_path = '/mnt/bn/ug-diffusion-yg-nas/guyiyang/data/revise_asr_parallel.csv'
    final_df.to_csv(final_output_file_path, index=False)

if __name__ == "__main__":
    main()