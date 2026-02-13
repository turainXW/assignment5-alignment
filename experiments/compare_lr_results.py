#!/usr/bin/env python3
"""
学习率扫描结果对比和分析
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

def load_experiment_logs(base_dir):
    """加载所有实验的日志"""
    base_path = Path(base_dir)
    experiments = {}

    for lr_dir in sorted(base_path.glob("lr_*")):
        lr_value = lr_dir.name.replace("lr_", "")
        log_file = lr_dir / "training_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                log = json.load(f)
                experiments[lr_value] = log
                print(f"加载实验: lr={lr_value}, steps={len(log['steps'])}")
        else:
            print(f"警告: 找不到日志文件 {log_file}")

    return experiments

def plot_comparison(experiments, output_dir):
    """绘制对比图表"""

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Learning Rate Sweep Comparison', fontsize=16, fontweight='bold')

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    # 1. Training Loss
    ax = axes[0, 0]
    for (lr, log), color in zip(sorted(experiments.items()), colors):
        if log['losses']:
            ax.plot(log['steps'], log['losses'], 'o-', label=f'lr={lr}',
                   color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Training Rewards
    ax = axes[0, 1]
    for (lr, log), color in zip(sorted(experiments.items()), colors):
        if log['train_rewards']:
            ax.plot(log['steps'], log['train_rewards'], 'o-', label=f'lr={lr}',
                   color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Train Reward', fontsize=12)
    ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Validation Rewards (最重要)
    ax = axes[0, 2]
    for (lr, log), color in zip(sorted(experiments.items()), colors):
        if log['val_rewards'] and log.get('val_steps'):
            ax.plot(log['val_steps'], log['val_rewards'], 'o-', label=f'lr={lr}',
                   color=color, alpha=0.7, linewidth=2, markersize=8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Validation Answer Rewards ⭐', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Target (25%)')

    # 4. Gradient Norm
    ax = axes[1, 0]
    for (lr, log), color in zip(sorted(experiments.items()), colors):
        if log.get('grad_norms'):
            ax.plot(log['steps'], log['grad_norms'], 'o-', label=f'lr={lr}',
                   color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 5. Token Entropy
    ax = axes[1, 1]
    for (lr, log), color in zip(sorted(experiments.items()), colors):
        if log.get('token_entropy'):
            ax.plot(log['steps'], log['token_entropy'], 'o-', label=f'lr={lr}',
                   color=color, alpha=0.7, linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Token Entropy', fontsize=12)
    ax.set_title('Token Entropy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 6. Final Validation Accuracy Bar Chart
    ax = axes[1, 2]
    lrs = []
    final_vals = []
    for lr, log in sorted(experiments.items()):
        if log['val_rewards']:
            lrs.append(lr)
            final_vals.append(log['val_rewards'][-1])

    bars = ax.bar(range(len(lrs)), final_vals, color=colors[:len(lrs)], alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels(lrs, rotation=45, ha='right')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Final Val Accuracy', fontsize=12)
    ax.set_title('Final Validation Accuracy', fontsize=14, fontweight='bold')
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Target (25%)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # 在柱状图上标注数值
    for i, (bar, val) in enumerate(zip(bars, final_vals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    output_path = Path(output_dir) / 'lr_sweep_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")

    return output_path

def print_summary(experiments):
    """打印实验总结"""
    print("\n" + "="*80)
    print("学习率扫描实验总结")
    print("="*80)

    results = []
    for lr, log in sorted(experiments.items()):
        if log['val_rewards']:
            final_val = log['val_rewards'][-1]
            max_val = max(log['val_rewards'])
            final_loss = log['losses'][-1] if log['losses'] else float('nan')
            final_train = log['train_rewards'][-1] if log['train_rewards'] else float('nan')

            # 检查是否达到25%目标
            reached_target = final_val >= 0.25

            results.append({
                'lr': lr,
                'final_val': final_val,
                'max_val': max_val,
                'final_loss': final_loss,
                'final_train': final_train,
                'reached_target': reached_target,
                'steps': len(log['steps'])
            })

    # 按最终验证准确率排序
    results.sort(key=lambda x: x['final_val'], reverse=True)

    print("\n按最终验证准确率排序:")
    print(f"{'排名':<6} {'学习率':<12} {'最终Val':<12} {'最大Val':<12} {'最终Loss':<12} {'达标':<6}")
    print("-" * 80)

    for i, r in enumerate(results, 1):
        target_mark = "✓" if r['reached_target'] else "✗"
        print(f"{i:<6} {r['lr']:<12} {r['final_val']:<12.4f} {r['max_val']:<12.4f} "
              f"{r['final_loss']:<12.6f} {target_mark:<6}")

    # 找出最佳学习率
    best = results[0]
    print("\n" + "="*80)
    print(f"🏆 最佳学习率: {best['lr']}")
    print(f"   最终验证准确率: {best['final_val']:.4f} ({best['final_val']*100:.2f}%)")
    print(f"   最大验证准确率: {best['max_val']:.4f} ({best['max_val']*100:.2f}%)")
    print(f"   是否达到25%目标: {'是 ✓' if best['reached_target'] else '否 ✗'}")
    print("="*80)

    # 分析趋势
    print("\n观察到的趋势:")

    # 1. 学习率与性能关系
    if len(results) >= 3:
        lr_values = [float(r['lr']) for r in results]
        val_values = [r['final_val'] for r in results]

        # 检查是否有明显的最优区间
        best_idx = val_values.index(max(val_values))
        if best_idx == 0:
            print("  • 最小学习率表现最好，可能需要尝试更小的学习率")
        elif best_idx == len(val_values) - 1:
            print("  • 最大学习率表现最好，可能需要尝试更大的学习率")
        else:
            print(f"  • 最优学习率在中间范围 ({results[best_idx]['lr']})")

    # 2. 梯度范数趋势
    print("\n  • 梯度范数分析:")
    for r in results[:3]:  # 只看前3个
        lr = r['lr']
        log = experiments[lr]
        if log.get('grad_norms'):
            avg_grad = np.mean(log['grad_norms'])
            std_grad = np.std(log['grad_norms'])
            print(f"    - lr={lr}: 平均梯度范数={avg_grad:.2f}, 标准差={std_grad:.2f}")

    # 3. Token Entropy趋势
    print("\n  • Token Entropy分析:")
    for r in results[:3]:
        lr = r['lr']
        log = experiments[lr]
        if log.get('token_entropy'):
            start_entropy = log['token_entropy'][0] if log['token_entropy'] else 0
            end_entropy = log['token_entropy'][-1] if log['token_entropy'] else 0
            entropy_change = end_entropy - start_entropy
            print(f"    - lr={lr}: 起始={start_entropy:.2f}, 结束={end_entropy:.2f}, "
                  f"变化={entropy_change:+.2f}")

    print("\n" + "="*80)

    return results

def main():
    parser = argparse.ArgumentParser(description='学习率扫描结果分析')
    parser.add_argument('base_dir', type=str, help='实验基础目录（包含lr_*子目录）')
    parser.add_argument('--no-plot', action='store_true', help='不生成图表')

    args = parser.parse_args()

    if not Path(args.base_dir).exists():
        print(f"错误: 找不到目录 {args.base_dir}")
        sys.exit(1)

    # 加载所有实验
    experiments = load_experiment_logs(args.base_dir)

    if not experiments:
        print("错误: 没有找到任何实验结果")
        sys.exit(1)

    # 打印总结
    results = print_summary(experiments)

    # 生成图表
    if not args.no_plot:
        plot_comparison(experiments, args.base_dir)

    # 保存总结到文件
    summary_path = Path(args.base_dir) / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Learning Rate Sweep Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Rank':<6} {'LR':<12} {'Final Val':<12} {'Max Val':<12} {'Target':<8}\n")
        f.write("-"*80 + "\n")
        for i, r in enumerate(results, 1):
            target_mark = "Yes" if r['reached_target'] else "No"
            f.write(f"{i:<6} {r['lr']:<12} {r['final_val']:<12.4f} {r['max_val']:<12.4f} {target_mark:<8}\n")
        f.write("\n")
        f.write(f"Best Learning Rate: {results[0]['lr']}\n")
        f.write(f"Best Final Val Accuracy: {results[0]['final_val']:.4f} ({results[0]['final_val']*100:.2f}%)\n")

    print(f"\n总结已保存到: {summary_path}")

if __name__ == '__main__':
    main()
