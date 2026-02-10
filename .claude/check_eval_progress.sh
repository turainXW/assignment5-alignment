#!/bin/bash
# 检查评估进度的脚本

echo "============================================"
echo "评估进程状态"
echo "============================================"
ps aux | grep evaluate_sft.py | grep -v grep | awk '{print "PID:", $2, "| CPU:", $3"%", "| Mem:", $4"%", "| GPU:", $NF}'
echo ""

echo "============================================"
echo "GPU使用情况"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

echo "============================================"
echo "评估日志最新进度"
echo "============================================"

if [ -f eval_results/size_128_log.txt ]; then
    echo "--- 模型1 (size_128) ---"
    tail -5 eval_results/size_128_log.txt 2>/dev/null || echo "日志文件为空或不存在"
    echo ""
fi

if [ -f eval_results/size_256_log.txt ]; then
    echo "--- 模型2 (size_256) ---"
    tail -5 eval_results/size_256_log.txt 2>/dev/null || echo "日志文件为空或不存在"
    echo ""
fi

if [ -f eval_results/size_512_log.txt ]; then
    echo "--- 模型3 (size_512) ---"
    tail -5 eval_results/size_512_log.txt 2>/dev/null || echo "日志文件为空或不存在"
    echo ""
fi

echo "============================================"
echo "结果文件状态"
echo "============================================"
ls -lh eval_results/*.json 2>/dev/null || echo "尚未生成结果文件"
echo ""
