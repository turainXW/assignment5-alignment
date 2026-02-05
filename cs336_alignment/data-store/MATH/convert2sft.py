import os
import json
from tqdm import tqdm

def process_directory(base_path, output_filename):
    """
    遍历指定目录下的所有子文件夹，提取 JSON 数据并保存为 jsonl
    """
    if not os.path.exists(base_path):
        print(f"跳过：找不到路径 {base_path}")
        return

    all_entries = []
    
    # os.walk 会递归进入所有文件夹（algebra, geometry, etc.）
    for root, dirs, files in os.walk(base_path):
        # 统计当前子文件夹
        category = os.path.basename(root)
        json_files = [f for f in files if f.endswith('.json')]
        
        if json_files:
            print(f"正在处理 {base_path} 中的 {category} ... ({len(json_files)} 个文件)")
            
        for file in json_files:
            file_path = os.path.join(root, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    
                    # 按照 SFT 标准格式提取
                    # 提示词模版建议和推理时保持一致
                    entry = {
                        "prompt": f"Question: {data.get('problem', '')}\n\nAnswer:",
                        "response": data.get('solution', ''),
                        "category": category # 保留类别，方便后续做评估分析
                    }
                    all_entries.append(entry)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    # 写入结果
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in all_entries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"成功导出 {len(all_entries)} 条数据到 {output_filename}\n")


def main():
    # --- 请确保这里的路径指向你解压后的 MATH 文件夹 ---
    # 假设你的当前目录下有一个名为 MATH 的文件夹
    math_root = "./MATH" 
    
    # 1. 处理训练集
    print("开始提取训练集...")
    process_directory("train", "math_train.jsonl")
    
    # 2. 处理测试集
    print("开始提取测试集...")
    process_directory( "test", "math_test.jsonl")

    print("所有处理已完成！")

if __name__ == "__main__":
    main()
