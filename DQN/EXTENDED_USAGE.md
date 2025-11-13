# DQN扩展数据集使用指南

## 概述

针对数据库中ConfigID超过1944条的情况（包含两种目标在新的六种环境下的图像参数遍历），对代码进行了扩展和优化。

## 主要改进

### 1. 灵活的目标筛选
支持四种目标筛选模式：
- `all`: 使用所有数据（不筛选目标）
- `manbo`: 仅使用满波目标的数据
- `panmao`: 仅使用盘帽目标的数据  
- `both`: 使用两种目标的数据（默认）

### 2. 两种缓存构建方法
- `database`: 基于数据库中实际存在的参数组合构建缓存（推荐）
- `theoretical`: 基于理论上所有可能的162种组合构建缓存

### 3. 智能参数编码
自动将数据库中的参数组合映射到我们的五元组格式，支持：
- 参数值的模糊匹配（处理浮点数精度问题）
- 无效组合的自动跳过
- 详细的进度和统计信息

## 使用方法

### 基本用法

```bash
# 使用数据库实际数据，筛选两种目标
python main.py --task complete --target-filter both --cache-method database

# 仅使用满波目标的数据
python main.py --task complete --target-filter manbo --cache-method database

# 使用所有数据（包括新增的环境数据）
python main.py --task complete --target-filter all --cache-method database
```

### 高级用法

```bash
# 强制重建缓存并训练1000轮
python main.py --task complete \
    --target-filter both \
    --cache-method database \
    --force-rebuild-cache \
    --episodes 1000 \
    --lr 0.0005

# 仅构建缓存查看数据分布
python main.py --task cache \
    --target-filter all \
    --cache-method database

# 使用预训练模型进行优化
python main.py --task optimize \
    --target-filter both \
    --cache-method database \
    --model-path ./models/best_model.pth
```

## 参数说明

### 新增参数

- `--target-filter`: 目标筛选模式
  - `all`: 使用所有ConfigID的数据
  - `manbo`: 仅满波目标 (ismanbo=1)
  - `panmao`: 仅盘帽目标 (ispanmao=1)  
  - `both`: 两种目标 (ismanbo=1 OR ispanmao=1)

- `--cache-method`: 缓存构建方法
  - `database`: 基于数据库实际数据（推荐，适用于扩展数据集）
  - `theoretical`: 理论上所有162种组合

### 原有参数
- `--task`: 执行任务 (cache/train/optimize/complete/exhaustive)
- `--episodes`: 训练轮数
- `--lr`: 学习率
- `--batch-size`: 批大小
- `--force-rebuild-cache`: 强制重建缓存

## 数据库兼容性

### 支持的数据结构
1. **扩展数据集**: ConfigID > 1944，包含新环境下的参数组合
2. **原始数据集**: ConfigID ≤ 1944，传统的参数组合
3. **混合数据集**: 同时包含原始和扩展数据

### 自动处理机制
- 自动检测数据库中的实际ConfigID范围
- 智能筛选有效的参数组合
- 跳过无法编码的异常组合
- 提供详细的数据统计信息

## 输出示例

```
[Cache] 从数据库获取到 2856 个参数组合
[Cache] 发现 2856 个可用的ConfigID
[Cache] ConfigID范围: 1 - 3888
[Cache] 进度: 50/2856 组合 -> ConfigID=127, F1=0.7234
...
缓存统计:
  总组合数: 2653
  有效组合数: 2341  
  最大F1分数: 0.8967
  平均F1分数: 0.6142
  构建时间: 45.2秒
```

## 性能优化建议

1. **首次运行**: 使用 `--cache-method database` 构建基于实际数据的缓存
2. **目标专注**: 根据研究目标选择合适的 `--target-filter`
3. **增量训练**: 保存检查点，支持断点续训
4. **资源管理**: 大数据集可能需要更多内存和训练时间

## 故障排除

### 常见问题

1. **缓存构建时间过长**
   - 检查数据库连接速度
   - 考虑使用更具体的target-filter减少数据量

2. **内存不足**
   - 减小batch-size
   - 使用更小的target-filter范围

3. **无效参数组合**
   - 检查数据库中的参数值是否在预定义范围内
   - 查看详细日志了解跳过的原因

4. **F1分数异常**
   - 验证evaluation表的数据质量
   - 检查JOIN条件是否正确

## 扩展建议

1. **新参数支持**: 修改参数空间定义以支持更多参数类型
2. **多目标优化**: 扩展为同时优化多个性能指标
3. **分布式训练**: 支持多GPU并行训练大规模数据集
4. **在线学习**: 支持动态添加新的参数组合数据
