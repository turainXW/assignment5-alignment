#!/bin/bash
# 并行评估三个SFT模型的脚本 - 使用正确的评估格式

# 激活虚拟环境
source .venv/bin/activate

# 输出目录
OUTPUT_DIR="eval_results_correct"
mkdir -p ${OUTPUT_DIR}

# 评估数据路径
EVAL_DATA="/root/assignment5-alignment/data/MATH/math_test.jsonl"

# 三个模型路径
MODEL1="sft_outputs/sft_ds_size_128_lr5e-05_bs2x8"
MODEL2="sft_outputs/sft_ds_size_256_lr5e-05_bs1x16"
MODEL3="sft_outputs/sft_ds_size_512_lr5e-05_bs1x16"

# 模型名称（用于输出文件）
NAME1="size_128"
NAME2="size_256"
NAME3="size_512"

echo "开始并行评估三个模型..."
echo "================================================"

# 使用不同的GPU并行评估三个模型
python3 cs336_alignment/evaluate_sft_correct.py \
    --model_path ${MODEL1} \
    --eval_data_path ${EVAL_DATA} \
    --max_samples 500 \
    --max_tokens 512 \
    --device cuda:0 \
    --gpu_memory_utilization 0.85 \
    --output_file ${OUTPUT_DIR}/${NAME1}_results.json \
    2>&1 | tee ${OUTPUT_DIR}/${NAME1}_log.txt &
PID1=$!

python3 cs336_alignment/evaluate_sft_correct.py \
    --model_path ${MODEL2} \
    --eval_data_path ${EVAL_DATA} \
    --max_samples 500 \
    --max_tokens 512 \
    --device cuda:1 \
    --gpu_memory_utilization 0.85 \
    --output_file ${OUTPUT_DIR}/${NAME2}_results.json \
    2>&1 | tee ${OUTPUT_DIR}/${NAME2}_log.txt &
PID2=$!

python3 cs336_alignment/evaluate_sft_correct.py \
    --model_path ${MODEL3} \
    --eval_data_path ${EVAL_DATA} \
    --max_samples 500 \
    --max_tokens 512 \
    --device cuda:2 \
    --gpu_memory_utilization 0.85 \
    --output_file ${OUTPUT_DIR}/${NAME3}_results.json \
    2>&1 | tee ${OUTPUT_DIR}/${NAME3}_log.txt &
PID3=$!

echo "模型1 (${NAME1}) 正在 GPU 0 上评估 (PID: ${PID1})"
echo "模型2 (${NAME2}) 正在 GPU 1 上评估 (PID: ${PID2})"
echo "模型3 (${NAME3}) 正在 GPU 2 上评估 (PID: ${PID3})"
echo "================================================"

# 等待所有进程完成
wait ${PID1}
echo "模型1 (${NAME1}) 评估完成"

wait ${PID2}
echo "模型2 (${NAME2}) 评估完成"

wait ${PID3}
echo "模型3 (${NAME3}) 评估完成"

echo "================================================"
echo "所有模型评估完成！"
echo ""
echo "正在生成汇总报告..."

# 创建汇总报告
python3 - <<EOF
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
models = [
    ("${NAME1}", "${MODEL1}"),
    ("${NAME2}", "${MODEL2}"),
    ("${NAME3}", "${MODEL3}"),
]

print("\n" + "="*70)
print("SFT模型评估结果汇总")
print("="*70)
print()

summary = []
for name, model_path in models:
    result_file = output_dir / f"{name}_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            results = json.load(f)

        accuracy = results['accuracy']
        format_accuracy = results['format_accuracy']
        correct = results['correct']
        has_boxed = results['has_boxed']
        total = results['total']

        print(f"模型: {name}")
        print(f"  路径: {model_path}")
        print(f"  答案正确率: {correct}/{total} = {accuracy:.2%}")
        print(f"  包含boxed答案: {has_boxed}/{total} = {format_accuracy:.2%}")
        print()

        summary.append({
            'model_name': name,
            'model_path': model_path,
            'accuracy': accuracy,
            'format_accuracy': format_accuracy,
            'correct': correct,
            'has_boxed': has_boxed,
            'total': total
        })
    else:
        print(f"警告: 未找到 {name} 的结果文件")
        print()

# 保存汇总
summary_file = output_dir / "summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("="*70)
print(f"汇总报告已保存到: {summary_file}")
print("="*70)
EOF

echo ""
echo "所有结果已保存到 ${OUTPUT_DIR}/ 目录"
echo "  - 每个模型的详细结果: ${OUTPUT_DIR}/<model>_results.json"
echo "  - 每个模型的日志: ${OUTPUT_DIR}/<model>_log.txt"
echo "  - 汇总报告: ${OUTPUT_DIR}/summary.json"
