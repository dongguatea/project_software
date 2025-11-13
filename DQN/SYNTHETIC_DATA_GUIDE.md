# DQN合成数据使用指南

## 概述

我已经为你的DQN图像参数优化系统添加了合成数据生成功能，可以在没有数据库的情况下进行DQN训练和测试。

## 新增功能

### 1. 合成F1数据生成
- 根据参数组合特征生成符合预期趋势的F1分数
- 支持自定义随机种子确保结果可重现
- 自动添加高斯噪声模拟真实数据的随机性

### 2. 数据趋势设定
按照你的要求设定了以下趋势：
- **LWIR传感器** (i_name=1) 比MWIR传感器表现更好
- **中温范围** (i_temp=1) 最优，过低/过高温度略差
- **分辨率排序**: 中分辨率 > 高分辨率 > 低分辨率
- **模糊/噪声**: 程度越高，F1分数越低
- **参数交互**: LWIR+中温+中分辨率组合有额外奖励

## 使用方法

### 方法1: 快速测试（推荐新手）

```bash
# 快速验证合成数据功能
python quick_test.py --task test

# 创建演示用缓存文件
python quick_test.py --task demo
```

### 方法2: 生成标准训练缓存

```bash
# 生成用于DQN训练的标准缓存
python generate_synthetic_data.py --task training --cache-path my_training_cache.pkl

# 验证生成的缓存文件
python generate_synthetic_data.py --task validate --cache-path my_training_cache.pkl
```

### 方法3: 完整DQN训练流程

```bash
# 使用合成数据运行完整DQN优化流程
python main.py --task complete --cache-method synthetic

# 指定缓存文件路径
python main.py --task complete --cache-method synthetic --cache-path my_cache.pkl

# 仅构建合成缓存
python main.py --task cache --cache-method synthetic

# 训练模型（使用合成数据）
python main.py --task train --cache-method synthetic --episodes 500
```

## 合成数据特点

### F1分数分布
- **范围**: 0.1 - 0.95
- **平均值**: 约0.75
- **最优组合**: 通常是LWIR + 中温 + 中分辨率 + 低模糊 + 低噪声

### 数据质量
- **组合数量**: 162个（2×3×3×3×3）
- **趋势明显**: 符合设定的参数影响规律
- **噪声适中**: 包含适量随机性但不掩盖主要趋势
- **全覆盖**: 包含所有可能的参数组合

## 输出示例

运行`python quick_test.py`后的典型输出：

```
=== 生成合成F1数据集 ===
[Synthetic] 生成了 162 个参数组合
[Synthetic] F1分数范围: 0.1234 - 0.9456
[Synthetic] 平均F1分数: 0.7543
[Synthetic] 最优组合 F1=0.9456:
[Synthetic]   name: Long Wave Infarared - LWIR
[Synthetic]   minTemperature: 0
[Synthetic]   maxTemperature: 60
[Synthetic]   hFOVPixels: 640
[Synthetic]   vFOVPixels: 512
[Synthetic]   percentBlur: 0.0
[Synthetic]   percentNoise: 0.0

=== 分析合成数据趋势 ===
[Analysis] MWIR平均F1: 0.7123
[Analysis] LWIR平均F1: 0.7963
[Analysis] 低温平均F1: 0.7234
[Analysis] 中温平均F1: 0.8012
[Analysis] 高温平均F1: 0.7343
...
```

## 优势

1. **无需数据库**: 可以在没有MySQL数据库的环境下测试DQN算法
2. **快速启动**: 几秒钟即可生成完整的训练数据
3. **趋势可控**: 数据符合预期的参数影响规律
4. **结果可重现**: 相同随机种子产生相同数据
5. **便于调试**: 可以快速验证DQN算法的有效性

## 与真实数据的区别

| 特性 | 合成数据 | 真实数据 |
|------|----------|----------|
| 生成速度 | 秒级 | 分钟级（需查询数据库） |
| 数据完整性 | 100%（所有162组合） | 取决于数据库内容 |
| 趋势一致性 | 完全符合设定规律 | 可能有异常值 |
| 真实性 | 模拟数据 | 实际测量结果 |
| 适用场景 | 算法测试、演示 | 实际应用 |

## 建议使用场景

1. **算法开发**: 验证DQN算法逻辑正确性
2. **参数调优**: 测试不同的训练超参数
3. **演示展示**: 快速展示系统功能
4. **教学培训**: 理解强化学习原理
5. **基准测试**: 建立性能基准线

## 注意事项

1. 合成数据仅供测试和演示，实际应用应使用真实数据
2. 如果要使用真实数据，请确保数据库配置正确
3. 可以通过修改`generate_synthetic_f1`函数调整数据生成规律
4. 建议在正式训练前先用合成数据验证代码正确性

## 下一步

完成合成数据测试后，你可以：
1. 使用合成数据训练DQN模型，验证算法收敛性
2. 调整训练超参数获得更好的性能
3. 将训练好的模型应用到真实数据上
4. 根据实际需求修改参数空间和奖励函数