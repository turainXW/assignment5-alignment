#!/bin/bash
# 实时监控 SFT 训练进度

TASK_OUTPUT="/tmp/claude-0/-root-assignment5-alignment/tasks/*.output"

clear
echo "====================================="
echo "  SFT Training Monitor (DeepSpeed)"
echo "====================================="
echo ""
echo "按 Ctrl+C 退出监控"
echo ""

while true; do
    clear
    echo "====================================="
    echo "  SFT Training Monitor"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "====================================="
    echo ""

    # GPU 状态
    echo "GPU Status:"
    echo "-------------------------------------"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s: %s | Mem: %s/%s MB | Util: %s%% | Temp: %s°C\n", $1, $2, $3, $4, $5, $6}'
    echo ""

    # 训练进程
    echo "Training Processes:"
    echo "-------------------------------------"
    if ps aux | grep -q "[p]ython.*sft_trainer_deepspeed"; then
        echo "✓ Training is running"
        ps aux | grep "[p]ython.*sft_trainer_deepspeed" | wc -l | xargs echo "  Active processes:"
    else
        echo "✗ No training process found"
    fi
    echo ""

    # 最新输出（查找最新的输出文件）
    LATEST_OUTPUT=$(ls -t /tmp/claude-0/-root-assignment5-alignment/tasks/*.output 2>/dev/null | head -1)

    if [ -n "$LATEST_OUTPUT" ]; then
        echo "Latest Training Output:"
        echo "-------------------------------------"
        tail -30 "$LATEST_OUTPUT" 2>&1 | \
            grep -v "Loading weights\|Materializing\|INFO.*config\|INFO.*llm_engine" | \
            grep -E "Epoch|loss|accuracy|Evaluating|Starting|✓|✗|Training|Loaded|Will evaluate|steps" | \
            tail -15
        echo ""
        echo "Full log: $LATEST_OUTPUT"
    fi

    echo ""
    echo "====================================="

    sleep 5
done
