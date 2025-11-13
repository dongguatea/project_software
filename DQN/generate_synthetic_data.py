# -*- coding: utf-8 -*-
"""
合成数据集生成器
用于创建DQN训练所需的合成F1缓存数据
"""

import os
import pickle
import argparse
from typing import Dict, Tuple

import numpy as np

from environment import ParameterSpace, create_synthetic_cache, analyze_synthetic_trends, exhaustive_search


def generate_multiple_datasets(base_seed: int = 42, num_datasets: int = 5, output_dir: str = './synthetic_data'):
    """生成多个不同随机种子的合成数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {}
    
    for i in range(num_datasets):
        seed = base_seed + i * 1000
        print(f"\n{'='*50}")
        print(f"生成数据集 {i+1}/{num_datasets} (seed={seed})")
        print(f"{'='*50}")
        
        cache_path = os.path.join(output_dir, f'synthetic_cache_seed_{seed}.pkl')
        cache = create_synthetic_cache(seed=seed, save_path=cache_path)
        
        # 分析趋势
        analyze_synthetic_trends(cache)
        
        # 找最优解
        best_combo, best_f1 = exhaustive_search(cache)
        param_space = ParameterSpace()
        best_params = param_space.decode_combo(best_combo)
        
        datasets[seed] = {
            'cache': cache,
            'best_combo': best_combo,
            'best_f1': best_f1,
            'best_params': best_params,
            'cache_path': cache_path
        }
        
        print(f"数据集 {i+1} 最优F1: {best_f1:.4f}")
    
    # 保存汇总信息
    summary_path = os.path.join(output_dir, 'datasets_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(datasets, f)
    
    # 生成汇总报告
    generate_summary_report(datasets, output_dir)
    
    return datasets


def generate_summary_report(datasets: Dict, output_dir: str):
    """生成数据集汇总报告"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("合成数据集汇总报告\n")
        f.write("="*50 + "\n\n")
        
        # 整体统计
        all_f1_values = []
        all_best_f1_values = []
        
        for seed, data in datasets.items():
            cache = data['cache']
            f1_values = list(cache.values())
            all_f1_values.extend(f1_values)
            all_best_f1_values.append(data['best_f1'])
            
            f.write(f"数据集 (seed={seed}):\n")
            f.write(f"  F1范围: {min(f1_values):.4f} - {max(f1_values):.4f}\n")
            f.write(f"  平均F1: {np.mean(f1_values):.4f}\n")
            f.write(f"  最优F1: {data['best_f1']:.4f}\n")
            f.write(f"  最优参数: {data['best_params']['name']}, ")
            f.write(f"温度({data['best_params']['minTemperature']}-{data['best_params']['maxTemperature']}), ")
            f.write(f"分辨率({data['best_params']['hFOVPixels']}x{data['best_params']['vFOVPixels']}), ")
            f.write(f"模糊({data['best_params']['percentBlur']:.1f}), ")
            f.write(f"噪声({data['best_params']['percentNoise']:.1f})\n\n")
        
        f.write("整体统计:\n")
        f.write(f"  数据集数量: {len(datasets)}\n")
        f.write(f"  每个数据集组合数: {len(next(iter(datasets.values()))['cache'])}\n")
        f.write(f"  总组合数: {len(all_f1_values)}\n")
        f.write(f"  整体F1范围: {min(all_f1_values):.4f} - {max(all_f1_values):.4f}\n")
        f.write(f"  整体平均F1: {np.mean(all_f1_values):.4f}\n")
        f.write(f"  整体F1标准差: {np.std(all_f1_values):.4f}\n")
        f.write(f"  最优F1范围: {min(all_best_f1_values):.4f} - {max(all_best_f1_values):.4f}\n")
        f.write(f"  平均最优F1: {np.mean(all_best_f1_values):.4f}\n")
    
    print(f"\n汇总报告已保存到: {report_path}")


def create_training_ready_cache(seed: int = 42, save_path: str = 'training_cache.pkl') -> str:
    """创建用于训练的标准缓存文件"""
    print("创建用于DQN训练的标准合成缓存...")
    
    cache = create_synthetic_cache(seed=seed, save_path=save_path)
    analyze_synthetic_trends(cache)
    
    # 验证缓存质量
    best_combo, best_f1 = exhaustive_search(cache)
    print(f"\n缓存质量验证:")
    print(f"  全局最优F1: {best_f1:.4f}")
    print(f"  缓存文件: {save_path}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")
    
    # 检查F1分布
    f1_values = list(cache.values())
    print(f"  F1分布统计:")
    print(f"    最小值: {min(f1_values):.4f}")
    print(f"    25%分位: {np.percentile(f1_values, 25):.4f}")
    print(f"    中位数: {np.percentile(f1_values, 50):.4f}")
    print(f"    75%分位: {np.percentile(f1_values, 75):.4f}")
    print(f"    最大值: {max(f1_values):.4f}")
    
    return save_path


def validate_cache_consistency(cache_path: str):
    """验证缓存文件的一致性"""
    print(f"验证缓存文件: {cache_path}")
    
    if not os.path.exists(cache_path):
        print("错误: 缓存文件不存在!")
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        
        print(f"  缓存加载成功")
        print(f"  包含 {len(cache)} 个参数组合")
        
        # 检查所有组合是否都是五元组
        param_space = ParameterSpace()
        expected_combos = set(param_space.all_combos)
        actual_combos = set(cache.keys())
        
        if expected_combos == actual_combos:
            print("  ✓ 所有预期的参数组合都存在")
        else:
            missing = expected_combos - actual_combos
            extra = actual_combos - expected_combos
            if missing:
                print(f"  ✗ 缺少 {len(missing)} 个组合")
            if extra:
                print(f"  ✗ 多出 {len(extra)} 个组合")
        
        # 检查F1值范围
        f1_values = list(cache.values())
        if all(0 <= f1 <= 1 for f1 in f1_values):
            print("  ✓ 所有F1值都在[0,1]范围内")
        else:
            invalid = [f1 for f1 in f1_values if not (0 <= f1 <= 1)]
            print(f"  ✗ 有 {len(invalid)} 个无效F1值")
        
        return True
        
    except Exception as e:
        print(f"  错误: 无法加载缓存文件 - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='合成数据集生成器')
    parser.add_argument('--task', type=str, default='single',
                       choices=['single', 'multiple', 'training', 'validate'],
                       help='任务类型: single=单个数据集, multiple=多个数据集, training=训练用缓存, validate=验证缓存')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--num-datasets', type=int, default=5,
                       help='生成数据集数量(multiple任务)')
    parser.add_argument('--output-dir', type=str, default='./synthetic_data',
                       help='输出目录')
    parser.add_argument('--cache-path', type=str, default='training_cache.pkl',
                       help='缓存文件路径')
    
    args = parser.parse_args()
    
    if args.task == 'single':
        print("生成单个合成数据集...")
        cache_path = os.path.join(args.output_dir, f'synthetic_cache_seed_{args.seed}.pkl')
        os.makedirs(args.output_dir, exist_ok=True)
        create_synthetic_cache(seed=args.seed, save_path=cache_path)
        
    elif args.task == 'multiple':
        print(f"生成 {args.num_datasets} 个合成数据集...")
        generate_multiple_datasets(
            base_seed=args.seed,
            num_datasets=args.num_datasets,
            output_dir=args.output_dir
        )
        
    elif args.task == 'training':
        print("创建用于DQN训练的标准缓存...")
        create_training_ready_cache(seed=args.seed, save_path=args.cache_path)
        
    elif args.task == 'validate':
        print("验证缓存文件...")
        validate_cache_consistency(args.cache_path)
    
    print("任务完成!")


if __name__ == "__main__":
    main()