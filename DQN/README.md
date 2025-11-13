# DQN图像参数优化系统

本项目是一个基于深度Q网络(DQN)的图像参数自动优化系统，用于在多场景、多目标条件下寻找最优的图像参数组合。

## 项目结构

```
DQN/
├── environment.py      # 模块1: 环境与奖励系统
├── dqn_agent.py       # 模块2: DQN网络与训练
├── optimizer.py       # 模块3: 参数优化搜索
├── main.py           # 主程序入口
├── requirements.txt  # 依赖包列表
└── README.md        # 说明文档
```

## 模块说明

### 模块1: 环境与奖励系统 (environment.py)
- **ParameterSpace**: 参数空间管理，支持162种参数组合
- **F1CacheManager**: F1性能缓存管理，支持数据库查询和缓存
- **ParamOptEnv**: 强化学习环境，实现状态-动作-奖励机制

### 模块2: DQN网络与训练 (dqn_agent.py)
- **QNetwork**: 深度Q网络结构
- **ReplayBuffer**: 经验回放缓冲区
- **DQNAgent**: DQN智能体，包含训练和推理逻辑
- **DQNTrainer**: DQN训练器，管理训练流程

### 模块3: 参数优化搜索 (optimizer.py)
- **ParameterOptimizer**: 参数优化器，使用训练好的DQN进行搜索
- **多种搜索策略**: 随机起点搜索、引导搜索、暴力搜索对比

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 完整流水线（推荐）
运行完整的优化流水线，包括缓存构建、模型训练和参数优化：

```bash
python main.py --task complete
```

### 2. 分步执行

#### 步骤1: 构建F1缓存
```bash
python main.py --task cache
```

#### 步骤2: 训练DQN模型
```bash
python main.py --task train --episodes 800
```

#### 步骤3: 参数优化搜索
```bash
python main.py --task optimize --model-path ./models/best_model.pth
```

### 3. 暴力搜索对比
```bash
python main.py --task exhaustive
```

## 配置参数

### 数据库配置
在 `environment.py` 中修改数据库连接配置：

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'config_db',
    'port': 3306
}
```

### 训练参数
```bash
python main.py --task complete \
    --episodes 1000 \
    --lr 0.001 \
    --batch-size 64 \
    --gamma 0.95
```

## 参数空间

系统支持以下图像参数的优化：

- **传感器类型**: MWIR, LWIR (2种)
- **温度范围**: (20,60), (0,70), (0,90) (3种)
- **分辨率**: (320,256), (640,512), (1024,1024) (3种)
- **模糊程度**: 0.0, 0.5, 1.0 (3种)
- **噪声水平**: 0.0, 0.5, 1.0 (3种)

总计: 2 × 3 × 3 × 3 × 3 = 162种组合

## 动作空间

DQN智能体支持10种离散动作：
- 动作0: 切换传感器类型
- 动作1-2: 调整温度档位
- 动作3-4: 调整分辨率档位
- 动作5-6: 调整模糊档位
- 动作7-8: 调整噪声档位
- 动作9: 无操作

## 输出结果

### 训练阶段输出
- 模型文件: `./models/best_model.pth`
- 训练统计: episode奖励、F1分数、损失等

### 优化阶段输出
- 优化结果: `./results/final_optimization_results.json`
- 包含最优参数组合、性能对比、动作分析等

### 示例输出
```
最终优化建议
==================================================
推荐方法: guided_search
最优F1分数: 0.8756
最优参数组合:
  name: Long Wave Infarared - LWIR
  minTemperature: 0
  maxTemperature: 70
  hFOVPixels: 640
  vFOVPixels: 512
  hFOVDeg: 20
  vFOVDeg: 16
  percentBlur: 0.0
  percentNoise: 0.0
```

## 系统特点

1. **模块化设计**: 三个独立模块，便于维护和扩展
2. **智能搜索**: 结合DQN的探索-利用机制，比暴力搜索更高效
3. **多策略优化**: 支持随机起点和引导搜索两种策略
4. **性能分析**: 提供与全局最优解的对比分析
5. **缓存机制**: 支持F1分数缓存，避免重复计算
6. **实验追踪**: 完整记录训练和优化过程

## 扩展建议

1. **优先经验回放**: 可以替换标准经验回放提升训练效率
2. **双DQN**: 减少Q值过估计问题
3. **多目标优化**: 扩展到同时优化多个性能指标
4. **分布式训练**: 支持多GPU训练加速
5. **超参数调优**: 使用Optuna等工具自动调优

## 注意事项

1. 确保数据库连接配置正确
2. 首次运行需要构建F1缓存，可能耗时较长
3. 训练过程建议使用GPU加速
4. 模型文件较大，注意存储空间

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库配置和网络连接
   - 确认数据库表结构正确

2. **内存不足**
   - 减小batch_size
   - 减少经验回放缓冲区大小

3. **训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查奖励函数设计

4. **GPU显存不足**
   - 减小网络规模
   - 使用CPU训练（修改device配置）
