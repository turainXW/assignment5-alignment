import json
import os
from pathlib import Path
from vllm import LLM, SamplingParams
import transformers
from transformers import Qwen2Tokenizer

# --- 1. 分词器强力补丁 (防止 AttributeError) ---
if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
    Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: [])

# --- 2. 导入奖励函数 ---
try:
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    from drgrpo_grader import r1_zero_reward_fn

def load_math_dataset_from_folder(base_path):
    """
    遍历文件夹下所有的 .json 文件并加载
    结构示例: MATH/test/algebra/1.json, 2.json...
    """
    data = []
    base_path = Path(base_path)
    # 使用 glob 递归查找所有 .json 文件
    json_files = list(base_path.glob("**/*.json"))
    
    print(f"在 {base_path} 中找到了 {len(json_files)} 个题目文件。")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                item = json.load(f)
                # 确保包含必要的字段
                if "problem" in item and "solution" in item:
                    data.append(item)
            except Exception as e:
                print(f"读取文件 {json_file} 出错: {e}")
    return data

def format_r1_zero_prompt(problem):
    # 严格遵循作业要求的 r1_zero 模板
    return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <thought> and <answer> tags, respectively.\n"
        f"User: {problem}\n"
        "Assistant: <thought>\n"
    )

def evaluate_vllm(vllm_model, reward_fn, dataset, eval_sampling_params, save_path):
    prompts = [format_r1_zero_prompt(ex['problem']) for ex in dataset]
    gold_answers = [ex['solution'] for ex in dataset]

    print(f"正在启动 vLLM 推理，共 {len(prompts)} 条数据...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    cats = {"cat1": 0, "cat2": 0, "cat3": 0}

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold = gold_answers[i]
        
        # 调用作业提供的评分函数
        score_dict = reward_fn(generated_text, gold)
        fmt = score_dict.get('format_reward', 0)
        ans = score_dict.get('answer_reward', 0)
        
        # 统计分类 (用于任务 b)
        if fmt == 1 and ans == 1:
            cats["cat1"] += 1
        elif fmt == 1 and ans == 0:
            cats["cat2"] += 1
        else: # fmt == 0
            cats["cat3"] += 1

        results.append({
            "problem": dataset[i]['problem'],
            "gold_solution": gold,
            "model_output": generated_text,
            "scores": score_dict
        })

    accuracy = cats["cat1"] / len(dataset) if len(dataset) > 0 else 0
    final_data = {
        "summary": {
            "total": len(dataset),
            "categories": cats,
            "accuracy": accuracy
        },
        "results": results
    }

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    return final_data

if __name__ == "__main__":
    # 路径配置
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    MODEL_PATH = str(project_root / "models" / "Qwen2.5-Math-1.5B")
    # 注意这里指向 test 文件夹
    DATA_DIR = str(project_root / "data" / "MATH" / "test")
    SAVE_PATH = str(project_root / "evaluate_math" / "zero_shot_results.json")

    # 初始化 vLLM
    llm = LLM(
        model=MODEL_PATH, 
        gpu_memory_utilization=0.8,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 加载并运行
    if os.path.exists(DATA_DIR):
        math_dataset = load_math_dataset_from_folder(DATA_DIR)
        if not math_dataset:
            print("错误：未能从文件夹中加载到任何题目。")
        else:
            print(f"成功加载 {len(math_dataset)} 道题目。")
            eval_results = evaluate_vllm(llm, r1_zero_reward_fn, math_dataset, sampling_params, SAVE_PATH)
            
            print("\n--- 评测总结 ---")
            print(f"准确率: {eval_results['summary']['accuracy']:.2%}")
            print(f"分类详情: {eval_results['summary']['categories']}")
            print(f"结果已保存至: {SAVE_PATH}")
    else:
        print(f"错误：找不到数据目录 {DATA_DIR}")
