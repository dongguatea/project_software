# -*- coding: utf-8 -*-
"""
主程序：DQN图像参数优化系统
整合三个模块的完整流水线
"""

import os
import sys
import argparse
import time
import pickle
from typing import Dict, Tuple

# 导入三个模块
from environment import F1CacheManager, ParamOptEnv, ParameterSpace, exhaustive_search
from dqn_agent import DQNAgent, DQNConfig, DQNTrainer
from optimizer import ParameterOptimizer, main_optimization_pipeline


class DQNOptimizationSystem:
    """DQN图像参数优化系统"""
    
    def __init__(self, 
                 cache_path: str = 'combo_f1_cache.pkl',
                 models_dir: str = './models',
                 results_dir: str = './results',
                 target_filter: str = "both"):  # 新增参数
        self.cache_path = cache_path
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.target_filter = target_filter  # "all", "manbo", "panmao", "both"
        
        # 创建目录
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化组件
        self.cache_manager = F1CacheManager(cache_path, target_filter)
        self.param_space = ParameterSpace()
        self.combo_f1 = None
        
    def step1_build_cache(self, force_rebuild: bool = False, method: str = "database") -> Dict:
        """
        步骤1: 构建或加载F1缓存
        
        Args:
            force_rebuild: 是否强制重建缓存
            method: 构建方法 - "database": 基于数据库实际数据, "theoretical": 理论上所有组合, "synthetic": 合成数据
        """
        print("=" * 60)
        print("步骤1: 构建F1性能缓存")
        print("=" * 60)
        print(f"目标筛选: {self.target_filter}")
        print(f"构建方法: {method}")
        
        start_time = time.time()
        
        # 构建缓存
        if method == "synthetic":
            from environment import create_synthetic_cache
            print("使用合成数据集...")
            self.combo_f1 = create_synthetic_cache(
                seed=42, 
                save_path=self.cache_path if not force_rebuild and not os.path.exists(self.cache_path) else None
            )
            if force_rebuild or not os.path.exists(self.cache_path):
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.combo_f1, f)
                print(f"合成缓存已保存到: {self.cache_path}")
        elif method == "database":
            self.combo_f1 = self.cache_manager.build_cache_from_db(use_cache=not force_rebuild)
        else:  # theoretical
            self.combo_f1 = self.cache_manager.build_cache_theoretical(use_cache=not force_rebuild)
        
        # 统计信息
        total_combos = len(self.combo_f1)
        valid_combos = sum(1 for f1 in self.combo_f1.values() if f1 > 0)
        max_f1 = max(self.combo_f1.values()) if self.combo_f1 else 0.0
        avg_f1 = sum(self.combo_f1.values()) / len(self.combo_f1) if self.combo_f1 else 0.0
        
        cache_stats = {
            'total_combinations': total_combos,
            'valid_combinations': valid_combos,
            'max_f1': max_f1,
            'average_f1': avg_f1,
            'cache_path': self.cache_path,
            'build_time': time.time() - start_time,
            'target_filter': self.target_filter,
            'method': method
        }
        
        print(f"缓存统计:")
        print(f"  总组合数: {total_combos}")
        print(f"  有效组合数: {valid_combos}")
        print(f"  最大F1分数: {max_f1:.4f}")
        print(f"  平均F1分数: {avg_f1:.4f}")
        print(f"  构建时间: {cache_stats['build_time']:.2f}秒")
        
        return cache_stats
    
    def step2_train_dqn(self, config: DQNConfig = None) -> Dict:
        """步骤2: 训练DQN模型"""
        print("\n" + "=" * 60)
        print("步骤2: 训练DQN模型")
        print("=" * 60)
        
        if self.combo_f1 is None:
            raise RuntimeError("请先运行步骤1构建缓存")
        
        # 使用默认配置或用户提供的配置
        if config is None:
            config = DQNConfig(
                max_episodes=800,
                epsilon_decay_steps=20000,
                train_start_size=1000,
                batch_size=64,
                lr=1e-3,
                gamma=0.95
            )
        
        # 创建环境
        env = ParamOptEnv(
            combo_f1=self.combo_f1,
            epsilon_gain=1e-3,
            patience=5,
            max_steps=50,
            seed=config.seed
        )
        
        # 创建智能体
        agent = DQNAgent(
            obs_dim=env.obs_dim(),
            n_actions=env.n_actions(),
            config=config
        )
        
        # 创建训练器
        trainer = DQNTrainer(agent, env, config)
        
        print(f"训练配置:")
        print(f"  最大训练轮数: {config.max_episodes}")
        print(f"  学习率: {config.lr}")
        print(f"  批大小: {config.batch_size}")
        print(f"  设备: {config.device}")
        
        # 开始训练
        training_results = trainer.train(save_dir=self.models_dir)
        
        print(f"\n训练完成！")
        print(f"最佳模型路径: {training_results['best_model_path']}")
        
        return training_results
    
    def step3_optimize_parameters(self, model_path: str = None) -> Dict:
        """步骤3: 使用训练好的模型进行参数优化"""
        print("\n" + "=" * 60)
        print("步骤3: 参数优化搜索")
        print("=" * 60)
        
        if self.combo_f1 is None:
            raise RuntimeError("请先运行步骤1构建缓存")
        
        # 确定模型路径
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"使用模型: {model_path}")
        
        # 执行优化流水线
        optimization_results = main_optimization_pipeline(
            model_path=model_path,
            combo_f1=self.combo_f1,
            output_dir=self.results_dir
        )
        
        return optimization_results
    
    def run_complete_pipeline(self, 
                            force_rebuild_cache: bool = False,
                            train_config: DQNConfig = None,
                            cache_method: str = "database") -> Dict:
        """运行完整的优化流水线"""
        print("开始DQN图像参数优化完整流水线")
        print("=" * 60)
        
        pipeline_results = {}
        
        # 步骤1: 构建缓存
        cache_stats = self.step1_build_cache(force_rebuild=force_rebuild_cache, method=cache_method)
        pipeline_results['cache_stats'] = cache_stats
        
        # 步骤2: 训练DQN
        training_results = self.step2_train_dqn(config=train_config)
        pipeline_results['training_results'] = training_results
        
        # 步骤3: 参数优化
        optimization_results = self.step3_optimize_parameters(
            model_path=training_results['best_model_path']
        )
        pipeline_results['optimization_results'] = optimization_results
        
        # 流水线总结
        print("\n" + "=" * 60)
        print("流水线执行完成")
        print("=" * 60)
        print(f"最优F1分数: {optimization_results['recommendation']['best_f1']:.4f}")
        print(f"最优参数组合:")
        for key, value in optimization_results['recommendation']['best_params'].items():
            print(f"  {key}: {value}")
        
        return pipeline_results
    
    def quick_exhaustive_search(self) -> Dict:
        """快速暴力搜索（用于对比验证）"""
        if self.combo_f1 is None:
            raise RuntimeError("请先运行步骤1构建缓存")
        
        print("执行暴力搜索获取全局最优解...")
        best_combo, best_f1 = exhaustive_search(self.combo_f1)
        best_params = self.param_space.decode_combo(best_combo)
        
        result = {
            'method': 'exhaustive_search',
            'best_combo': best_combo,
            'best_f1': best_f1,
            'best_params': best_params
        }
        
        print(f"全局最优F1: {best_f1:.4f}")
        print(f"全局最优参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DQN图像参数优化系统')
    parser.add_argument('--task', type=str, default='complete',
                       choices=['cache', 'train', 'optimize', 'complete', 'exhaustive'],
                       help='执行任务: cache=构建缓存, train=训练模型, optimize=参数优化, complete=完整流水线, exhaustive=暴力搜索')
    
    parser.add_argument('--cache-path', type=str, default='combo_f1_cache.pkl',
                       help='F1缓存文件路径')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='模型保存目录')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='结果保存目录')
    parser.add_argument('--model-path', type=str, default=None,
                       help='预训练模型路径（用于optimize任务）')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=800,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='折扣因子')
    
    # 其他选项
    parser.add_argument('--force-rebuild-cache', action='store_true',
                       help='强制重建缓存')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--target-filter', type=str, default='both',
                       choices=['all', 'manbo', 'panmao', 'both'],
                       help='目标筛选: all=所有数据, manbo=仅满波, panmao=仅盘帽, both=两种目标')
    parser.add_argument('--cache-method', type=str, default='synthetic',
                       choices=['database', 'theoretical', 'synthetic'],
                       help='缓存构建方法: database=基于数据库实际数据, theoretical=理论上所有组合, synthetic=合成数据集')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = DQNOptimizationSystem(
        cache_path=args.cache_path,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        target_filter=args.target_filter
    )
    
    # 创建训练配置
    train_config = DQNConfig(
        max_episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed
    )
    
    try:
        if args.task == 'cache':
            system.step1_build_cache(force_rebuild=args.force_rebuild_cache, method=args.cache_method)
        
        elif args.task == 'train':
            system.step1_build_cache(force_rebuild=args.force_rebuild_cache, method=args.cache_method)
            system.step2_train_dqn(config=train_config)
        
        elif args.task == 'optimize':
            system.step1_build_cache(force_rebuild=False, method=args.cache_method)
            system.step3_optimize_parameters(model_path=args.model_path)
        
        elif args.task == 'complete':
            system.run_complete_pipeline(
                force_rebuild_cache=args.force_rebuild_cache,
                train_config=train_config,
                cache_method=args.cache_method
            )
        
        elif args.task == 'exhaustive':
            system.step1_build_cache(force_rebuild=args.force_rebuild_cache, method=args.cache_method)
            system.quick_exhaustive_search()
        
        print("\n任务执行完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
