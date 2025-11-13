# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证DQN系统的合成数据功能
"""

import os
import sys

# 添加DQN模块到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import create_synthetic_cache, ParamOptEnv, ParameterSpace, exhaustive_search, analyze_synthetic_trends
from generate_synthetic_data import create_training_ready_cache, validate_cache_consistency


def quick_test():
    """快速测试合成数据功能"""
    print("DQN合成数据快速测试")
    print("=" * 50)
    
    # 1. 生成合成缓存
    print("\n1. 生成合成F1缓存...")
    cache_path = 'quick_test_cache.pkl'
    synthetic_cache = create_synthetic_cache(seed=42, save_path=cache_path)
    
    # 2. 分析数据趋势
    print("\n2. 分析数据趋势...")
    analyze_synthetic_trends(synthetic_cache)
    
    # 3. 找全局最优
    print("\n3. 暴力搜索全局最优...")
    best_combo, best_f1 = exhaustive_search(synthetic_cache)
    param_space = ParameterSpace()
    best_params = param_space.decode_combo(best_combo)
    
    print(f"全局最优F1: {best_f1:.4f}")
    print("全局最优参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 4. 测试DQN环境
    print("\n4. 测试DQN环境交互...")
    env = ParamOptEnv(combo_f1=synthetic_cache, seed=42)
    
    # 测试多个episode
    total_improvements = []
    for episode in range(5):
        obs = env.reset(random_start=True)
        start_f1 = env._f1(env.state)
        
        print(f"\nEpisode {episode+1}:")
        print(f"  起始F1: {start_f1:.4f}")
        print(f"  起始参数: {env.decode_current_combo()['name']}")
        
        # 执行一些步骤
        for step in range(15):
            # 简单的贪心策略：尝试所有动作，选择最好的
            best_action = 0
            best_next_f1 = -1
            
            current_state = env.state.copy()
            current_f1 = env._f1(current_state)
            
            for action in range(env.n_actions()):
                # 模拟执行动作
                next_state = env._apply_action(current_state, action)
                next_f1 = env._f1(next_state)
                
                if next_f1 > best_next_f1:
                    best_next_f1 = next_f1
                    best_action = action
            
            # 执行最佳动作
            obs, reward, done, info = env.step(best_action)
            
            if step % 5 == 0:
                print(f"    步骤 {step+1}: F1={info['f1']:.4f}, 动作={info['action_name']}")
            
            if done:
                print(f"    提前终止于步骤 {step+1}: {info['termination_reason']}")
                break
        
        final_f1 = info['f1']
        improvement = final_f1 - start_f1
        total_improvements.append(improvement)
        
        print(f"  结束F1: {final_f1:.4f}")
        print(f"  改进幅度: {improvement:+.4f}")
    
    # 5. 总结测试结果
    print(f"\n5. 测试总结:")
    print(f"  平均改进: {sum(total_improvements)/len(total_improvements):+.4f}")
    print(f"  最大改进: {max(total_improvements):+.4f}")
    print(f"  改进成功率: {sum(1 for x in total_improvements if x > 0) / len(total_improvements):.2%}")
    print(f"  与全局最优的差距: {best_f1 - max([start_f1 + imp for start_f1, imp in zip([env._f1([0,0,0,0,0])]*5, total_improvements)]):.4f}")
    
    # 6. 验证缓存文件
    print(f"\n6. 验证缓存文件...")
    validate_cache_consistency(cache_path)
    
    # 清理测试文件
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"清理测试文件: {cache_path}")
    
    print(f"\n✅ 快速测试完成！合成数据功能正常工作。")


def create_demo_cache():
    """创建演示用的缓存文件"""
    print("创建演示用的合成缓存文件...")
    
    cache_path = 'demo_synthetic_cache.pkl'
    create_training_ready_cache(seed=42, save_path=cache_path)
    
    print(f"✅ 演示缓存已创建: {cache_path}")
    print(f"现在可以使用以下命令测试完整DQN流程:")
    print(f"python main.py --task complete --cache-method synthetic --cache-path {cache_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN合成数据快速测试')
    parser.add_argument('--task', type=str, default='test',
                       choices=['test', 'demo'],
                       help='任务类型: test=快速测试, demo=创建演示缓存')
    
    args = parser.parse_args()
    
    if args.task == 'test':
        quick_test()
    elif args.task == 'demo':
        create_demo_cache()