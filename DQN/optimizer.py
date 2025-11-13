# -*- coding: utf-8 -*-
"""
模块3：训练后自动寻优
使用训练好的DQN模型进行参数优化搜索
"""

import os
import random
import json
import time
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch

from environment import ParamOptEnv, F1CacheManager, ParameterSpace, exhaustive_search
from dqn_agent import DQNAgent, DQNConfig


class ParameterOptimizer:
    """参数优化器：使用训练好的DQN进行参数搜索"""
    
    def __init__(self, model_path: str, combo_f1: Dict[Tuple[int,int,int,int,int], float]):
        self.model_path = model_path
        self.combo_f1 = combo_f1
        self.param_space = ParameterSpace()
        
        # 创建环境（用于评估）
        self.env = ParamOptEnv(combo_f1=combo_f1, seed=42)
        
        # 加载训练好的模型
        self.agent = self._load_trained_agent()
        
        
    
    def _load_trained_agent(self) -> DQNAgent:
        """加载训练好的DQN智能体"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        # 创建智能体
        obs_dim = self.env.obs_dim()
        n_actions = self.env.n_actions()
        agent = DQNAgent(obs_dim, n_actions, config)
        
        # 加载模型
        agent.load_model(self.model_path)
        
        print(f"成功加载训练好的模型: {self.model_path}")
        print(f"模型训练步数: {agent.total_steps}")
        print(f"模型训练轮数: {agent.episode_count}")
        
        return agent
    
    def greedy_search_from_random_starts(self, 
                                       n_trials: int = 100, 
                                       max_steps: int = 50,
                                       verbose: bool = True) -> Dict:
        """从多个随机起点进行贪心搜索"""
        results = []
        best_f1 = -float('inf')
        best_combo = None
        best_trajectory = None
        
        if verbose:
            print(f"开始从 {n_trials} 个随机起点进行贪心搜索...")
        
        for trial in range(n_trials):
            result = self._single_greedy_search(max_steps=max_steps, random_start=True)
            results.append(result)
            
            if result['final_f1'] > best_f1:
                best_f1 = result['final_f1']
                best_combo = result['final_combo']
                best_trajectory = result['trajectory']
            
            if verbose and (trial + 1) % 20 == 0:
                print(f"进度: {trial + 1}/{n_trials}, 当前最佳F1: {best_f1:.4f}")
        
        summary = {
            'method': 'greedy_search_from_random_starts',
            'n_trials': n_trials,
            'best_f1': best_f1,
            'best_combo': best_combo,
            'best_params': self.param_space.decode_combo(best_combo),
            'best_trajectory': best_trajectory,
            'all_results': results,
            'success_rate': sum(1 for r in results if r['improvement'] > 0) / len(results),
            'avg_improvement': np.mean([r['improvement'] for r in results]),
            'avg_steps': np.mean([r['steps'] for r in results])
        }
        
        if verbose:
            print(f"\n贪心搜索完成！")
            print(f"最佳F1分数: {best_f1:.4f}")
            print(f"成功改进率: {summary['success_rate']:.2%}")
            print(f"平均改进幅度: {summary['avg_improvement']:.4f}")
        
        return summary
    
    def guided_search_from_best_combos(self, 
                                     top_k: int = 20, 
                                     max_steps: int = 30,
                                     verbose: bool = True) -> Dict:
        """从数据库中的最佳组合开始引导搜索"""
        # 找到top-k最佳组合作为起点
        sorted_combos = sorted(self.combo_f1.items(), key=lambda x: x[1], reverse=True)
        top_combos = [combo for combo, f1 in sorted_combos[:top_k]]
        
        results = []
        best_f1 = -float('inf')
        best_combo = None
        best_trajectory = None
        
        if verbose:
            print(f"从前 {top_k} 个最佳组合开始引导搜索...")
        
        for i, start_combo in enumerate(top_combos):
            result = self._single_greedy_search(
                max_steps=max_steps, 
                start_combo=start_combo,
                random_start=False
            )
            results.append(result)
            
            if result['final_f1'] > best_f1:
                best_f1 = result['final_f1']
                best_combo = result['final_combo']
                best_trajectory = result['trajectory']
            
            if verbose and (i + 1) % 5 == 0:
                print(f"进度: {i + 1}/{top_k}, 当前最佳F1: {best_f1:.4f}")
        
        summary = {
            'method': 'guided_search_from_best_combos',
            'top_k': top_k,
            'best_f1': best_f1,
            'best_combo': best_combo,
            'best_params': self.param_space.decode_combo(best_combo),
            'best_trajectory': best_trajectory,
            'all_results': results,
            'improvement_rate': sum(1 for r in results if r['improvement'] > 0) / len(results),
            'avg_improvement': np.mean([r['improvement'] for r in results]),
            'avg_steps': np.mean([r['steps'] for r in results])
        }
        
        if verbose:
            print(f"\n引导搜索完成！")
            print(f"最佳F1分数: {best_f1:.4f}")
            print(f"改进率: {summary['improvement_rate']:.2%}")
        
        return summary
    
    def _single_greedy_search(self, 
                            max_steps: int = 50,
                            start_combo: Optional[Tuple[int,int,int,int,int]] = None,
                            random_start: bool = True) -> Dict:
        """单次贪心搜索"""
        # 重置环境
        if random_start:
            obs = self.env.reset(random_start=True)
        else:
            obs = self.env.reset(random_start=False)
            if start_combo is not None:
                self.env.state = list(start_combo)
                obs = self.env._encode_obs(self.env.state)
        
        start_f1 = self.env._f1(self.env.state)
        start_combo_actual = self.env.current_combo_tuple()
        trajectory = [start_combo_actual]
        
        # 贪心执行
        for step in range(max_steps):
            # 使用训练好的智能体选择动作（贪心模式）
            action = self.agent.select_action(obs, eval_mode=True)
            obs, reward, done, info = self.env.step(action)
            trajectory.append(self.env.current_combo_tuple())
            
            if done:
                break
        
        final_f1 = self.env._f1(self.env.state)
        final_combo = self.env.current_combo_tuple()
        
        return {
            'start_combo': start_combo_actual,
            'start_f1': start_f1,
            'final_combo': final_combo,
            'final_f1': final_f1,
            'improvement': final_f1 - start_f1,
            'steps': len(trajectory) - 1,
            'trajectory': trajectory,
            'terminated_early': step < max_steps - 1
        }
    
    def compare_with_exhaustive(self) -> Dict:
        """与暴力搜索结果进行对比"""
        print("执行暴力搜索获取全局最优解...")
        
        # 暴力搜索
        global_best_combo, global_best_f1 = exhaustive_search(self.combo_f1)
        
        # DQN搜索
        print("执行DQN贪心搜索...")
        dqn_result = self.greedy_search_from_random_starts(n_trials=100, verbose=False)
        
        # 比较结果
        gap = global_best_f1 - dqn_result['best_f1']
        gap_percentage = (gap / global_best_f1) * 100 if global_best_f1 > 0 else 0
        
        comparison = {
            'global_optimum': {
                'combo': global_best_combo,
                'params': self.param_space.decode_combo(global_best_combo),
                'f1': global_best_f1
            },
            'dqn_result': {
                'combo': dqn_result['best_combo'],
                'params': dqn_result['best_params'],
                'f1': dqn_result['best_f1']
            },
            'performance_gap': gap,
            'performance_gap_percentage': gap_percentage,
            'is_global_optimum': gap < 1e-6,
            'dqn_efficiency': {
                'success_rate': dqn_result['success_rate'],
                'avg_steps': dqn_result['avg_steps']
            }
        }
        
        print(f"\n=== 性能对比结果 ===")
        print(f"全局最优F1: {global_best_f1:.4f}")
        print(f"DQN最优F1: {dqn_result['best_f1']:.4f}")
        print(f"性能差距: {gap:.4f} ({gap_percentage:.2f}%)")
        print(f"是否达到全局最优: {'是' if comparison['is_global_optimum'] else '否'}")
        
        return comparison
    
    def analyze_action_patterns(self, n_episodes: int = 50) -> Dict:
        """分析DQN智能体的动作模式"""
        action_counts = {i: 0 for i in range(self.env.n_actions())}
        action_rewards = {i: [] for i in range(self.env.n_actions())}
        
        for episode in range(n_episodes):
            obs = self.env.reset(random_start=True)
            
            for step in range(50):
                action = self.agent.select_action(obs, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                
                action_counts[action] += 1
                action_rewards[action].append(reward)
                
                if done:
                    break
        
        # 统计分析
        analysis = {
            'action_frequencies': action_counts,
            'action_percentages': {k: v/sum(action_counts.values())*100 
                                 for k, v in action_counts.items()},
            'action_avg_rewards': {k: np.mean(v) if v else 0.0 
                                 for k, v in action_rewards.items()},
            'action_names': self.env.action_names,
            'most_used_action': max(action_counts, key=action_counts.get),
            'most_rewarding_action': max(action_rewards, 
                                       key=lambda k: np.mean(action_rewards[k]) if action_rewards[k] else -float('inf'))
        }
        
        print(f"\n=== 动作模式分析 ===")
        for i, name in enumerate(self.env.action_names):
            freq = analysis['action_percentages'][i]
            avg_reward = analysis['action_avg_rewards'][i]
            print(f"动作 {i} ({name}): 频率 {freq:.1f}%, 平均奖励 {avg_reward:+.4f}")
        
        return analysis


def save_results(results: Dict, filepath: str):
    """保存结果到文件"""
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_json = convert_numpy(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {filepath}")


def main_optimization_pipeline(model_path: str, 
                             combo_f1: Dict[Tuple[int,int,int,int,int], float],
                             output_dir: str = './optimization_results') -> Dict:
    """主要的优化流水线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建优化器
    optimizer = ParameterOptimizer(model_path, combo_f1)
    env = optimizer.env
    
    # 1. 随机起点贪心搜索
    print("=== 阶段1: 随机起点贪心搜索 ===")
    random_search_results = optimizer.greedy_search_from_random_starts(n_trials=100)
    save_results(random_search_results, os.path.join(output_dir, 'random_search_results.json'))
    
    # 2. 引导搜索
    print("\n=== 阶段2: 从最佳组合开始引导搜索 ===")
    guided_search_results = optimizer.guided_search_from_best_combos(top_k=20)
    save_results(guided_search_results, os.path.join(output_dir, 'guided_search_results.json'))
    
    # 3. 与暴力搜索对比
    print("\n=== 阶段3: 与全局最优解对比 ===")
    comparison_results = optimizer.compare_with_exhaustive()
    save_results(comparison_results, os.path.join(output_dir, 'comparison_results.json'))
    
    # 4. 动作模式分析
    print("\n=== 阶段4: 动作模式分析 ===")
    action_analysis = optimizer.analyze_action_patterns(n_episodes=50)
    save_results(action_analysis, os.path.join(output_dir, 'action_analysis.json'))
    
    # 整合最终结果
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': model_path,
        'random_search': random_search_results,
        'guided_search': guided_search_results,
        'comparison': comparison_results,
        'action_analysis': action_analysis,
        'recommendation': {
            'best_method': 'guided_search' if guided_search_results['best_f1'] >= random_search_results['best_f1'] else 'random_search',
            'best_f1': max(guided_search_results['best_f1'], random_search_results['best_f1']),
            'best_combo': guided_search_results['best_combo'] if guided_search_results['best_f1'] >= random_search_results['best_f1'] else random_search_results['best_combo'],
            'best_params': guided_search_results['best_params'] if guided_search_results['best_f1'] >= random_search_results['best_f1'] else random_search_results['best_params']
        }
    }
    
    save_results(final_results, os.path.join(output_dir, 'final_optimization_results.json'))
    
    # 输出最终建议
    print(f"\n{'='*50}")
    print(f"最终优化建议")
    print(f"{'='*50}")
    print(f"推荐方法: {final_results['recommendation']['best_method']}")
    print(f"最优F1分数: {final_results['recommendation']['best_f1']:.4f}")
    print(f"最优参数组合:")
    for key, value in final_results['recommendation']['best_params'].items():
        print(f"  {key}: {value}")
    print(f"结果保存目录: {output_dir}")
    
    return final_results


if __name__ == "__main__":
    # 测试优化模块
    print("测试优化模块...")
    
    # 创建模拟数据用于测试
    param_space = ParameterSpace()
    mock_cache = {}
    
    # 生成模拟F1分数（某些组合更优）
    for i, combo in enumerate(param_space.all_combos):
        # 使某些特定组合具有更高的F1分数
        base_score = random.uniform(0.3, 0.7)
        if combo[0] == 1:  # LWIR传感器
            base_score += 0.1
        if combo[1] == 1:  # 中等温度范围
            base_score += 0.05
        if combo[2] == 1:  # 中等分辨率
            base_score += 0.05
        mock_cache[combo] = min(base_score, 0.95)  # 限制最大值
    
    # 设置一个明确的全局最优解
    global_best_combo = (1, 1, 1, 0, 0)  # LWIR, 中温, 中分辨率, 低模糊, 低噪声
    mock_cache[global_best_combo] = 0.95
    
    print(f"创建了 {len(mock_cache)} 个组合的模拟缓存")
    print(f"全局最优组合: {param_space.decode_combo(global_best_combo)}")
    print(f"全局最优F1: {mock_cache[global_best_combo]:.4f}")
    
    # 注意：这里需要一个预训练的模型文件
    # 在实际使用中，应该先运行训练模块生成模型
    print("\n注意: 实际使用需要先训练DQN模型")
    print("请先运行训练模块生成模型文件，然后使用以下代码:")
    print("""
    # 示例使用代码:
    model_path = './models/best_model.pth'
    results = main_optimization_pipeline(model_path, combo_f1_cache)
    """)
    
    print("\n优化模块测试完成！")
