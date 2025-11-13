# -*- coding: utf-8 -*-
"""
DQN for Discrete Image-Parameter Optimization
- Single-file: environment + DQN + training + search
- Author: your friendly RL assistant
"""

import os
import sys
import math
import json
import time
import pickle
import random
import argparse
import configparser
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mysql.connector
except Exception:
    mysql = None


# =========================
# 1) 参数档位定义
# =========================
NAMES = ['Mid-wave Infarared - MWIR', 'Long Wave Infarared - LWIR']  
MIN_MAX_TEMPERATURE_OPTIONS = [(0, 30), (0, 50), (0, 70)]           
FOVPIXEL_OPTIONS = [(320, 256), (640, 512), (1024, 1024)]            
SENSIM_BLUR_CHOICES = np.linspace(0, 1, 3).tolist()                   
SENSIM_NOISE_CHOICES = np.linspace(0, 1, 3).tolist()                  

# 视场角 = 像素 / 32
def fov_deg_from_pixels(px: int) -> int:
    return int(round(px / 32))

# 全部组合枚举（162种）
ALL_COMBOS = []
for i_name, name in enumerate(NAMES):
    for i_temp, (mn, mx) in enumerate(MIN_MAX_TEMPERATURE_OPTIONS):
        for i_fov, (hpix, vpix) in enumerate(FOVPIXEL_OPTIONS):
            for i_blur, blur in enumerate(SENSIM_BLUR_CHOICES):
                for i_noise, noise in enumerate(SENSIM_NOISE_CHOICES):
                    ALL_COMBOS.append((i_name, i_temp, i_fov, i_blur, i_noise))


def load_db_config(config_path: str = None) -> Dict[str, any]:
    """从config.ini文件加载数据库配置"""
    if config_path is None:
        # 默认在database目录下查找config.ini
        config_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'config.ini')
    
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在 {config_path}，使用默认配置")
        return {
            'host': 'localhost',
            'user': 'root',
            'password': 'your_password',
            'database': 'config_db',
            'port': 3306
        }
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        
        if 'mysqld' not in config:
            print("警告: 配置文件中没有[mysqld]节，使用默认配置")
            return {
                'host': 'localhost',
                'user': 'root',
                'password': 'your_password',
                'database': 'config_db',
                'port': 3306
            }
        
        db_config = {
            'host': config.get('mysqld', 'host', fallback='localhost'),
            'user': config.get('mysqld', 'user', fallback='root'),
            'password': config.get('mysqld', 'password', fallback='your_password'),
            'database': config.get('mysqld', 'database', fallback='config_db'),
            'port': config.getint('mysqld', 'port', fallback=3306)
        }
        
        return db_config
        
    except Exception as e:
        print(f"警告: 读取配置文件失败 {e}，使用默认配置")
        return {
            'host': 'localhost',
            'user': 'root',
            'password': 'your_password',
            'database': 'config_db',
            'port': 3306
        }


# =========================
# 2) 数据库配置（从config.ini读取）
# =========================
DB_CONFIG = load_db_config()

PARAM_TABLE = 'param_sets'
IMAGES_TABLE = 'images'
EVAL_TABLE_CANDIDATES = ['evaluation', 'evalution']   # 你之前提到过可能拼写是 evalution
CACHE_PATH = 'combo_f1_cache.pkl'


# =========================
# 3) 预计算：组合 -> 综合F1
# =========================
def get_eval_table_name(conn) -> str:
    """Try possible eval table names."""
    cur = conn.cursor()
    for name in EVAL_TABLE_CANDIDATES:
        try:
            cur.execute(f"SELECT 1 FROM `{name}` LIMIT 1")
            cur.fetchall()
            cur.close()
            return name
        except Exception:
            continue
    cur.close()
    raise RuntimeError("找不到 evaluation/evalution 表，请检查表名。")

def avg_f1_for_combo(conn, eval_table: str,
                     name: str, minT: int, maxT: int,
                     hpix: int, vpix: int, blur: float, noise: float) -> float:
    """
    计算该图像参数组合在 两个目标 × 6 场景 的综合F1（直接 join 所有匹配 ConfigID）
    """
    hdeg = fov_deg_from_pixels(hpix)
    vdeg = fov_deg_from_pixels(vpix)

    # 通过 param_sets 精确匹配图像参数（不限定场景），且 (ismanbo=1 OR ispanmao=1)
    # 再 join images -> evaluation 取平均 F1
    # 注意：MySQL 5.7，无CTE，直接三表join
    sql = f"""
    SELECT AVG(e.f1)
    FROM `{PARAM_TABLE}` p
    JOIN `{IMAGES_TABLE}` i ON i.ConfigID = p.ConfigID
    JOIN `{eval_table}` e ON e.image_id = i.image_id
    WHERE p.name = %s
      AND p.minTemperature = %s
      AND p.maxTemperature = %s
      AND p.hFOVPixels = %s
      AND p.vFOVPixels = %s
      AND p.hFOVDeg = %s
      AND p.vFOVDeg = %s
      AND ABS(p.percentBlur - %s) < 1e-9
      AND ABS(p.percentNoise - %s) < 1e-9
      AND (p.ismanbo = 1 OR p.ispanmao = 1)
    """
    cur = conn.cursor()
    cur.execute(sql, (name, minT, maxT, hpix, vpix, hdeg, vdeg, blur, noise))
    row = cur.fetchone()
    cur.close()
    if row is None or row[0] is None:
        return 0.0
    return float(row[0])

def build_combo_f1_cache(use_cache=True) -> Dict[Tuple[int,int,int,int,int], float]:
    """
    遍历162种参数组合，计算综合F1，存入缓存。
    键是 (i_name, i_temp, i_fov, i_blur, i_noise) 五元组。
    """
    if use_cache and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            print(f"[Cache] 从 {CACHE_PATH} 载入综合F1缓存。")
            return pickle.load(f)

    if mysql is None:
        raise RuntimeError("未安装 mysql-connector-python，请先安装。")
    conn = mysql.connector.connect(**DB_CONFIG)
    eval_table = get_eval_table_name(conn)

    mapping = {}
    total = len(ALL_COMBOS)
    t0 = time.time()
    for idx, (i_name, i_temp, i_fov, i_blur, i_noise) in enumerate(ALL_COMBOS, 1):
        name = NAMES[i_name]
        mn, mx = MIN_MAX_TEMPERATURE_OPTIONS[i_temp]
        hpix, vpix = FOVPIXEL_OPTIONS[i_fov]
        blur = SENSIM_BLUR_CHOICES[i_blur]
        noise = SENSIM_NOISE_CHOICES[i_noise]
        f1 = avg_f1_for_combo(conn, eval_table, name, mn, mx, hpix, vpix, blur, noise)
        mapping[(i_name, i_temp, i_fov, i_blur, i_noise)] = f1
        if idx % 10 == 0 or idx == total:
            print(f"[Precompute] {idx}/{total} 组合 -> F1={f1:.4f}")
    conn.close()

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"[Cache] 已保存到 {CACHE_PATH}，用时 {time.time()-t0:.1f}s")
    return mapping


# =========================
# 4) 环境：Gym-like（离散动作 + 查表奖励）
# =========================
class ParamOptEnv:
    """
    状态：五个离散索引 (i_name, i_temp, i_fov, i_blur, i_noise)
    观测：one-hot 向量（2 + 3 + 3 + 3 + 3 = 14）
    动作：10个
        0: toggle name
        1: temp_prev, 2: temp_next
        3: fov_prev,  4: fov_next
        5: blur_prev, 6: blur_next
        7: noise_prev,8: noise_next
        9: no-op
    奖励：r = F1(new) - F1(old)
    终止：最近 patience 步的增益都 < epsilon_gain，或步数超过 max_steps
    """
    def __init__(self, combo_f1: Dict[Tuple[int,int,int,int,int], float],
                 epsilon_gain: float = 1e-3,
                 patience: int = 5,
                 max_steps: int = 50,
                 seed: int = 42):
        self.combo_f1 = combo_f1
        self.epsilon_gain = epsilon_gain
        self.patience = patience
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    @staticmethod
    def obs_dim() -> int:
        return 2 + 3 + 3 + 3 + 3  # 14

    @staticmethod
    def n_actions() -> int:
        return 10

    def _encode_obs(self, state) -> np.ndarray:
        i_name, i_temp, i_fov, i_blur, i_noise = state
        v = []
        # name one-hot (2)
        for k in range(2): v.append(1.0 if k == i_name else 0.0)
        # temp one-hot (3)
        for k in range(3): v.append(1.0 if k == i_temp else 0.0)
        # fov one-hot (3)
        for k in range(3): v.append(1.0 if k == i_fov else 0.0)
        # blur one-hot (3)
        for k in range(3): v.append(1.0 if k == i_blur else 0.0)
        # noise one-hot (3)
        for k in range(3): v.append(1.0 if k == i_noise else 0.0)
        return np.array(v, dtype=np.float32)

    def _f1(self, state) -> float:
        return self.combo_f1.get(tuple(state), 0.0)

    def _apply_action(self, state, action):
        i_name, i_temp, i_fov, i_blur, i_noise = state
        if action == 0:
            i_name = 1 - i_name
        elif action == 1:
            i_temp = max(0, i_temp - 1)
        elif action == 2:
            i_temp = min(2, i_temp + 1)
        elif action == 3:
            i_fov = max(0, i_fov - 1)
        elif action == 4:
            i_fov = min(2, i_fov + 1)
        elif action == 5:
            i_blur = max(0, i_blur - 1)
        elif action == 6:
            i_blur = min(2, i_blur + 1)
        elif action == 7:
            i_noise = max(0, i_noise - 1)
        elif action == 8:
            i_noise = min(2, i_noise + 1)
        elif action == 9:
            pass  # no-op
        return [i_name, i_temp, i_fov, i_blur, i_noise]

    def reset(self, random_start=True):
        if random_start:
            self.state = [
                self.rng.randint(0, 1),
                self.rng.randint(0, 2),
                self.rng.randint(0, 2),
                self.rng.randint(0, 2),
                self.rng.randint(0, 2),
            ]
        else:
            self.state = [0, 0, 0, 0, 0]
        self.step_cnt = 0
        self.recent_gains = deque(maxlen=self.patience)
        obs = self._encode_obs(self.state)
        return obs

    def step(self, action: int):
        assert 0 <= action < self.n_actions()
        old_f1 = self._f1(self.state)
        new_state = self._apply_action(self.state, action)
        new_f1 = self._f1(new_state)
        reward = new_f1 - old_f1

        self.state = new_state
        self.step_cnt += 1
        self.recent_gains.append(abs(reward))

        done = False
        # 平台期：最近若干步增益均很小
        if len(self.recent_gains) == self.patience and max(self.recent_gains) < self.epsilon_gain:
            done = True
        if self.step_cnt >= self.max_steps:
            done = True

        obs = self._encode_obs(self.state)
        info = {"f1": new_f1}
        return obs, reward, done, info

    def current_combo_tuple(self) -> Tuple[int,int,int,int,int]:
        return tuple(self.state)

    def decode_combo(self, combo_tuple):
        i_name, i_temp, i_fov, i_blur, i_noise = combo_tuple
        name = NAMES[i_name]
        minT, maxT = MIN_MAX_TEMPERATURE_OPTIONS[i_temp]
        hpix, vpix = FOVPIXEL_OPTIONS[i_fov]
        blur = SENSIM_BLUR_CHOICES[i_blur]
        noise = SENSIM_NOISE_CHOICES[i_noise]
        return {
            "name": name,
            "minTemperature": minT,
            "maxTemperature": maxT,
            "hFOVPixels": hpix,
            "vFOVPixels": vpix,
            "hFOVDeg": fov_deg_from_pixels(hpix),
            "vFOVDeg": fov_deg_from_pixels(vpix),
            "percentBlur": blur,
            "percentNoise": noise,
        }


# =========================
# 5) DQN 组件
# =========================
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


@dataclass
class DQNConfig:
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    train_start_size: int = 1000
    target_update_every: int = 500
    max_episodes: int = 800
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.q = QNet(obs_dim, n_actions).to(cfg.device)
        self.tgt = QNet(obs_dim, n_actions).to(cfg.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

        self.buffer = ReplayBuffer()
        self.total_steps = 0

    def epsilon(self):
        # linear decay
        eps = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
              max(0.0, (self.cfg.epsilon_decay_steps - self.total_steps) / self.cfg.epsilon_decay_steps)
        return eps

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(self.cfg.device)
            qvals = self.q(x)
            a = int(torch.argmax(qvals, dim=1).item())
        return a

    def optimize(self):
        if len(self.buffer) < self.cfg.train_start_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.from_numpy(s).float().to(self.cfg.device)
        a = torch.from_numpy(a).long().to(self.cfg.device)
        r = torch.from_numpy(r).float().to(self.cfg.device)
        ns = torch.from_numpy(ns).float().to(self.cfg.device)
        d = torch.from_numpy(d).float().to(self.cfg.device)

        with torch.no_grad():
            next_q = self.tgt(ns)  # [B, A]
            max_next_q = next_q.max(dim=1)[0]
            y = r + self.cfg.gamma * (1.0 - d) * max_next_q

        qvals = self.q(s)
        q_a = qvals.gather(1, a.view(-1, 1)).squeeze(1)

        loss = F.smooth_l1_loss(q_a, y)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.optim.step()

        if self.total_steps % self.cfg.target_update_every == 0:
            self.tgt.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save({
            'q': self.q.state_dict(),
            'cfg': self.cfg.__dict__,
            'total_steps': self.total_steps
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.q.load_state_dict(ckpt['q'])
        self.tgt.load_state_dict(self.q.state_dict())
        self.total_steps = ckpt.get('total_steps', 0)


# =========================
# 6) 训练循环 & 寻优
# =========================
def train(env: ParamOptEnv, agent: DQNAgent, out_dir: str = './runs'):
    os.makedirs(out_dir, exist_ok=True)
    best_f1 = -1.0
    best_combo = None
    best_path = os.path.join(out_dir, 'dqn_best.pt')

    for ep in range(agent.cfg.max_episodes):
        obs = env.reset(random_start=True)
        ep_reward = 0.0
        last_info_f1 = env._f1(env.state)
        done = False
        steps = 0
        while not done:
            a = agent.select_action(obs)
            next_obs, r, done, info = env.step(a)
            agent.buffer.push(obs, a, r, next_obs, float(done))
            agent.total_steps += 1
            agent.optimize()
            obs = next_obs
            ep_reward += r
            steps += 1
            last_info_f1 = info['f1']

        # 以 episode 末 F1 作为该条轨迹代表性能
        if last_info_f1 > best_f1:
            best_f1 = last_info_f1
            best_combo = env.current_combo_tuple()
            agent.save(best_path)

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[Train] Ep{ep+1:04d} steps={steps} epR={ep_reward:+.4f} eps={agent.epsilon():.3f} "
                  f"lastF1={last_info_f1:.4f} | bestF1={best_f1:.4f} {best_combo}")

    print(f"[Train] 完成。bestF1={best_f1:.4f} 组合={best_combo}")
    return best_combo, best_f1, best_path


def greedy_search(env: ParamOptEnv, agent: DQNAgent, start_random_trials: int = 50, max_steps: int = 50):
    """从若干随机起点出发贪心执行，取最好F1"""
    best_f1 = -1.0
    best_combo = None
    for t in range(start_random_trials):
        obs = env.reset(random_start=True)
        # eval 模式：直接用贪心（不含随机）
        for _ in range(max_steps):
            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0).to(agent.cfg.device)
                qvals = agent.q(x)
                a = int(torch.argmax(qvals, dim=1).item())
            obs, r, done, info = env.step(a)
            if done:
                break
        f1 = info['f1']
        if f1 > best_f1:
            best_f1 = f1
            best_combo = env.current_combo_tuple()
    return best_combo, best_f1


def exhaustive_best_from_cache(combo_f1: Dict[Tuple[int,int,int,int,int], float]):
    """暴力从缓存查表找全局最优（162种，瞬间完成）"""
    best = max(combo_f1.items(), key=lambda kv: kv[1])
    return best  # ((tuple), f1)


# =========================
# 7) CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='precompute',
                        choices=['precompute', 'train', 'search', 'exhaustive'],
                        help='precompute: 预计算缓存; train: 训练DQN; search: 训练后贪心寻优; exhaustive: 直接查表最优')
    parser.add_argument('--runs', type=str, default='./runs', help='模型输出目录')
    parser.add_argument('--episodes', type=int, default=800, help='训练episode数量')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gain_eps', type=float, default=1e-3)
    args = parser.parse_args()

    if args.task == 'precompute':
        build_combo_f1_cache(use_cache=False)
        return

    # 载入缓存
    combo_f1 = build_combo_f1_cache(use_cache=True)

    # 创建环境
    env = ParamOptEnv(combo_f1=combo_f1,
                      epsilon_gain=args.gain_eps,
                      patience=args.patience,
                      max_steps=50,
                      seed=args.seed)

    if args.task == 'exhaustive':
        (combo, f1) = exhaustive_best_from_cache(combo_f1)
        print("[Exhaustive] 最优组合：", env.decode_combo(combo))
        print("[Exhaustive] 最优综合F1：", f1)
        return

    # 构建Agent
    cfg = DQNConfig(max_episodes=args.episodes, seed=args.seed)
    agent = DQNAgent(obs_dim=env.obs_dim(), n_actions=env.n_actions(), cfg=cfg)

    best_path = os.path.join(args.runs, 'dqn_best.pt')

    if args.task == 'train':
        best_combo, best_f1, best_path = train(env, agent, out_dir=args.runs)
        print("[Train] 最优组合：", env.decode_combo(best_combo))
        print("[Train] 最优综合F1：", best_f1)
        print("[Train] 模型已保存：", best_path)
        return

    if args.task == 'search':
        # 加载已训练模型
        agent.load(best_path)
        combo, f1 = greedy_search(env, agent, start_random_trials=100, max_steps=50)
        print("[Search] 贪心寻优得到组合：", env.decode_combo(combo))
        print("[Search] 对应综合F1：", f1)
        return


if __name__ == '__main__':
    main()
