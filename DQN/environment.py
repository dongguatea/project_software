# -*- coding: utf-8 -*-
"""
模块1：环境与奖励系统
基于数据库/缓存的综合F1查表环境
"""

import os
import sys
import time
import pickle
import random
import configparser
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

try:
    import mysql.connector
except ImportError:
    mysql = None


# =========================
# 参数档位定义
# =========================
NAMES = ['Mid-wave Infarared - MWIR', 'Long Wave Infarared - LWIR']
MIN_MAX_TEMPERATURE_OPTIONS = [(0, 30), (0, 60), (0, 90)]
FOVPIXEL_OPTIONS = [(320, 256), (640, 512), (1024, 1024)]
SENSIM_BLUR_CHOICES = np.linspace(0, 1, 3).tolist()  # [0.0, 0.5, 1.0]
SENSIM_NOISE_CHOICES = np.linspace(0, 1, 3).tolist()  # [0.0, 0.5, 1.0]


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
            'password': 'asd515359',
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
                'password': 'asd515359',
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
        
        # for sec in config.sections():
        #     if sec.lower().strip() == 'mysqld' :
        #         db_config = {k.lower() : v.strip() for k,v in config[sec].items()}
        #         for key in ('host','port','user','password','database'):
        #             if key not in db_config:
        #                 raise ValueError(f"[{sec}] 缺少必要配置项：{key}")
        #         db_config["port"] = int(db_config["port"])
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


# 从配置文件加载数据库配置
configpath = Path(__file__).parent/ 'config.ini'
DB_CONFIG = load_db_config(configpath)

PARAM_TABLE = 'param_sets'
IMAGES_TABLE = 'images'
EVAL_TABLE_CANDIDATES = 'evalution'
CACHE_PATH = 'combo_f1_cache.pkl'


def fov_deg_from_pixels(px: int) -> int:
    """将像素转换为视场角"""
    return int(round(px / 32))


class ParameterSpace:
    """参数空间管理类"""
    """
    采用五元组 (i_name, i_temp, i_fov, i_blur, i_noise) 表示参数组合，i_是各个参数的索引
    """
    
    def __init__(self):
        self.all_combos = self._generate_all_combos()
        #获取所有参数组合的长度
        self.n_combos = len(self.all_combos)
    
    def _generate_all_combos(self) -> List[Tuple[int, int, int, int, int]]:
        """生成所有参数组合"""
        combos = []
        # i_name,i_temp,i_fov这些是参数的索引，将所有的参数索引保存在all_combos中
        for i_name, name in enumerate(NAMES):
            for i_temp, (mn, mx) in enumerate(MIN_MAX_TEMPERATURE_OPTIONS):
                for i_fov, (hpix, vpix) in enumerate(FOVPIXEL_OPTIONS):
                    for i_blur, blur in enumerate(SENSIM_BLUR_CHOICES):
                        for i_noise, noise in enumerate(SENSIM_NOISE_CHOICES):
                            combos.append((i_name, i_temp, i_fov, i_blur, i_noise))
        return combos
    
    def decode_combo(self, combo_tuple: Tuple[int, int, int, int, int]) -> Dict:
        """将参数索引解码为实际参数值"""
        #combo_tuple是一个五元组，包含了各个参数的索引
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


class F1CacheManager:
    """F1缓存管理器 - 支持扩展的数据集"""
    
    def __init__(self, cache_path: str = CACHE_PATH, target_filter: str = "all"):
        self.cache_path = cache_path
        self.param_space = ParameterSpace()
        self.target_filter = target_filter  # "all", "manbo", "panmao", "both"
    
    def get_eval_table_name(self, conn) -> str:
        """尝试找到正确的评估表名"""
        cur = conn.cursor()
        
        try:
            cur.execute(f"SELECT 1 FROM `{EVAL_TABLE_CANDIDATES}` LIMIT 1")
            cur.fetchall()
            cur.close()
            return EVAL_TABLE_CANDIDATES
        except Exception:
            pass
        cur.close()
        raise RuntimeError("找不到 evalution 表，请检查表名。")
    
    def _get_target_condition(self) -> str:
        """根据目标筛选条件生成SQL WHERE子句"""
        if self.target_filter == "manbo":
            return "AND p.ismanbo = 1"
        elif self.target_filter == "panmao":
            return "AND p.ispanmao = 1" 
        elif self.target_filter == "both":
            return "AND (p.ismanbo = 1 OR p.ispanmao = 1)"
        else:  # "all"
            return ""  # 不添加目标筛选条件
    
    def get_available_configs(self, conn) -> List[int]:
        """获取数据库中所有可用的ConfigID"""
        cur = conn.cursor()
        target_condition = self._get_target_condition()
        
        sql = f"""
        SELECT DISTINCT p.ConfigID 
        FROM `{PARAM_TABLE}` p
        JOIN `{IMAGES_TABLE}` i ON i.ConfigID = p.ConfigID
        WHERE 1=1 {target_condition}
        ORDER BY p.ConfigID
        """
        
        cur.execute(sql)
        config_ids = [row[0] for row in cur.fetchall()]
        cur.close()
        
        print(f"[Cache] 发现 {len(config_ids)} 个可用的ConfigID")
        print(f"[Cache] ConfigID范围: {min(config_ids)} - {max(config_ids)}")
        
        return config_ids
    
    def get_param_combinations_from_db(self, conn) -> List[Tuple[Dict, int]]:
        """从数据库获取所有实际存在的参数组合"""
        cur = conn.cursor()
        target_condition = self._get_target_condition()
        
        sql = f"""
        SELECT DISTINCT 
            p.name, p.minTemperature, p.maxTemperature,
            p.hFOVPixels, p.vFOVPixels, p.hFOVDeg, p.vFOVDeg,
            p.percentBlur, p.percentNoise, p.ConfigID
        FROM `{PARAM_TABLE}` p
        JOIN `{IMAGES_TABLE}` i ON i.ConfigID = p.ConfigID
        WHERE 1=1 {target_condition}
        ORDER BY p.ConfigID
        """
        
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        
        combinations = []
        for row in rows:
            params = {
                "name": row[0],
                "minTemperature": row[1],
                "maxTemperature": row[2],
                "hFOVPixels": row[3],
                "vFOVPixels": row[4],
                "hFOVDeg": row[5],
                "vFOVDeg": row[6],
                "percentBlur": row[7],
                "percentNoise": row[8]
            }
            config_id = row[9]
            combinations.append((params, config_id))
        
        print(f"[Cache] 从数据库获取到 {len(combinations)} 个参数组合")
        return combinations
    
    def encode_params_to_combo(self, params: Dict) -> Optional[Tuple[int,int,int,int,int]]:
        """将数据库参数编码为五元组格式"""
        try:
            # 查找name对应的索引
            if params["name"] not in NAMES:
                return None
            i_name = NAMES.index(params["name"])
            
            # 查找温度范围对应的索引
            temp_tuple = (params["minTemperature"], params["maxTemperature"])
            if temp_tuple not in MIN_MAX_TEMPERATURE_OPTIONS:
                return None
            i_temp = MIN_MAX_TEMPERATURE_OPTIONS.index(temp_tuple)
            
            # 查找分辨率对应的索引
            fov_tuple = (params["hFOVPixels"], params["vFOVPixels"])
            if fov_tuple not in FOVPIXEL_OPTIONS:
                return None
            i_fov = FOVPIXEL_OPTIONS.index(fov_tuple)
            
            # 查找模糊程度对应的索引
            blur = params["percentBlur"]
            blur_distances = [abs(blur - choice) for choice in SENSIM_BLUR_CHOICES]
            i_blur = blur_distances.index(min(blur_distances))
            
            # 查找噪声程度对应的索引
            noise = params["percentNoise"]
            noise_distances = [abs(noise - choice) for choice in SENSIM_NOISE_CHOICES]
            i_noise = noise_distances.index(min(noise_distances))
            
            return (i_name, i_temp, i_fov, i_blur, i_noise)
            
        except (ValueError, IndexError):
            return None
    
    def avg_f1_for_combo(self, conn, eval_table: str,
                         name: str, minT: int, maxT: int,
                         hpix: int, vpix: int, blur: float, noise: float) -> float:
        """计算指定参数组合的综合F1分数"""
        hdeg = fov_deg_from_pixels(hpix)
        vdeg = fov_deg_from_pixels(vpix)
        target_condition = self._get_target_condition()
        
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
          {target_condition}
        """
        
        cur = conn.cursor()
        cur.execute(sql, (name, minT, maxT, hpix, vpix, hdeg, vdeg, blur, noise))
        row = cur.fetchone()
        cur.close()
        
        if row is None or row[0] is None:
            return 0.0
        return float(row[0])
    
    def build_cache_from_db(self, use_cache: bool = True) -> Dict[Tuple[int,int,int,int,int], float]:
        """从数据库构建F1缓存 - 基于实际存在的数据"""
        if use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                print(f"[Cache] 从 {self.cache_path} 载入综合F1缓存。")
                return pickle.load(f)
        
        if mysql is None:
            raise RuntimeError("未安装 mysql-connector-python，请先安装。")
        
        print("[Cache] 开始从数据库构建F1缓存...")
        conn = mysql.connector.connect(**DB_CONFIG)
        eval_table = self.get_eval_table_name(conn)
        
        # 获取数据库中实际存在的参数组合
        db_combinations = self.get_param_combinations_from_db(conn)
        
        mapping = {}
        total = len(db_combinations)
        t0 = time.time()
        
        for idx, (params, config_id) in enumerate(db_combinations, 1):
            # 将数据库参数编码为五元组
            combo = self.encode_params_to_combo(params)
            
            if combo is None:
                print(f"[Cache] 警告: ConfigID {config_id} 的参数组合无法编码，跳过")
                continue
            
            # 计算F1分数
            f1 = self.avg_f1_for_combo(
                conn, eval_table,
                params["name"], params["minTemperature"], params["maxTemperature"],
                params["hFOVPixels"], params["vFOVPixels"],
                params["percentBlur"], params["percentNoise"]
            )
            
            mapping[combo] = f1
            
            if idx % 50 == 0 or idx == total:
                print(f"[Cache] 进度: {idx}/{total} 组合 -> ConfigID={config_id}, F1={f1:.4f}")
        
        conn.close()
        
        # 保存缓存
        with open(self.cache_path, 'wb') as f:
            pickle.dump(mapping, f)
        
        print(f"[Cache] 缓存构建完成，用时 {time.time()-t0:.1f}s")
        print(f"[Cache] 成功缓存 {len(mapping)} 个参数组合，已保存到 {self.cache_path}")
        return mapping
    
    def build_cache(self, use_cache: bool = True) -> Dict[Tuple[int,int,int,int,int], float]:
        """构建或加载F1缓存 - 兼容旧版本的方法"""
        return self.build_cache_from_db(use_cache)
    
    def build_cache_theoretical(self, use_cache: bool = True) -> Dict[Tuple[int,int,int,int,int], float]:
        """构建理论上所有可能组合的F1缓存（原版本方法）"""
        if use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                print(f"[Cache] 从 {self.cache_path} 载入综合F1缓存。")
                return pickle.load(f)
        
        if mysql is None:
            raise RuntimeError("未安装 mysql-connector-python，请先安装。")
        
        print("[Cache] 开始构建理论F1缓存...")
        conn = mysql.connector.connect(**DB_CONFIG)
        eval_table = self.get_eval_table_name(conn)
        
        mapping = {}
        total = len(self.param_space.all_combos)
        t0 = time.time()
        
        for idx, combo in enumerate(self.param_space.all_combos, 1):
            #combo是一个五元组，包含了各个参数的索引
            i_name, i_temp, i_fov, i_blur, i_noise = combo
            params = self.param_space.decode_combo(combo)
            
            f1 = self.avg_f1_for_combo(
                conn, eval_table,
                params["name"], params["minTemperature"], params["maxTemperature"],
                params["hFOVPixels"], params["vFOVPixels"],
                params["percentBlur"], params["percentNoise"]
            )
            
            mapping[combo] = f1
            
            if idx % 20 == 0 or idx == total:
                print(f"[Cache] 进度: {idx}/{total} 组合 -> F1={f1:.4f}")
        
        conn.close()
        
        # 保存缓存
        with open(self.cache_path, 'wb') as f:
            pickle.dump(mapping, f)
        
        print(f"[Cache] 缓存构建完成，用时 {time.time()-t0:.1f}s，已保存到 {self.cache_path}")
        return mapping


class ParamOptEnv:
    """
    图像参数优化环境,强化学习的环境
    多维状态空间的强化学习环境，可以采用各个离散索引作为状态表示，以每个动作的变化作为离散动作空间
    状态：五个离散索引 (i_name, i_temp, i_fov, i_blur, i_noise)
    观测：one-hot 向量（2 + 3 + 3 + 3 + 3 = 14维）
    动作：10个离散动作
        0: 切换传感器类型 (toggle name)
        1: 降低温度档位 (temp_prev)
        2: 提高温度档位 (temp_next)
        3: 降低分辨率档位 (fov_prev)
        4: 提高分辨率档位 (fov_next)
        5: 降低模糊档位 (blur_prev)
        6: 提高模糊档位 (blur_next)
        7: 降低噪声档位 (noise_prev)
        8: 提高噪声档位 (noise_next)
        9: 无操作 (no-op)
    奖励：增量式F1奖励 r = F1(new) - F1(old)
    终止：连续patience步的增益都小于epsilon_gain，或步数超过max_steps
    """
    
    def __init__(self, 
                 combo_f1: Dict[Tuple[int,int,int,int,int], float],
                 epsilon_gain: float = 1e-3,
                 patience: int = 5,
                 max_steps: int = 50,
                 seed: int = 42):
        #f1缓存了所有参数组合的F1分数
        self.combo_f1 = combo_f1
        self.epsilon_gain = epsilon_gain
        self.patience = patience
        self.max_steps = max_steps
        self.param_space = ParameterSpace()
        self.rng = random.Random(seed)
        
        # 动作名称映射（用于调试和可视化）
        self.action_names = [
            "切换传感器类型", "降低温度档位", "提高温度档位",
            "降低分辨率档位", "提高分辨率档位", "降低模糊档位",
            "提高模糊档位", "降低噪声档位", "提高噪声档位", "无操作"
        ]
        
        self.reset()
    
    @staticmethod
    def obs_dim() -> int:
        """观测向量维度"""
        return 2 + 3 + 3 + 3 + 3  # 14
    
    @staticmethod
    def n_actions() -> int:
        """动作空间大小"""
        return 10
    
    def _encode_obs(self, state: List[int]) -> np.ndarray:
        """将状态编码为one-hot观测向量"""
        i_name, i_temp, i_fov, i_blur, i_noise = state
        v = []
        
        # 传感器类型 one-hot (2维)
        for k in range(2): 
            v.append(1.0 if k == i_name else 0.0)
        
        # 温度档位 one-hot (3维)
        for k in range(3): 
            v.append(1.0 if k == i_temp else 0.0)
        
        # 分辨率档位 one-hot (3维)
        for k in range(3): 
            v.append(1.0 if k == i_fov else 0.0)
        
        # 模糊档位 one-hot (3维)
        for k in range(3): 
            v.append(1.0 if k == i_blur else 0.0)
        
        # 噪声档位 one-hot (3维)
        for k in range(3): 
            v.append(1.0 if k == i_noise else 0.0)
        
        return np.array(v, dtype=np.float32)
    
    def _f1(self, state: List[int]) -> float:
        """获取状态对应的F1分数"""
        return self.combo_f1.get(tuple(state), 0.0)
    
    def _apply_action(self, state: List[int], action: int) -> List[int]:
        """应用动作到状态"""
        i_name, i_temp, i_fov, i_blur, i_noise = state
        
        if action == 0:  # 切换传感器类型
            i_name = 1 - i_name
        elif action == 1:  # 降低温度档位
            i_temp = max(0, i_temp - 1)
        elif action == 2:  # 提高温度档位
            i_temp = min(2, i_temp + 1)
        elif action == 3:  # 降低分辨率档位
            i_fov = max(0, i_fov - 1)
        elif action == 4:  # 提高分辨率档位
            i_fov = min(2, i_fov + 1)
        elif action == 5:  # 降低模糊档位
            i_blur = max(0, i_blur - 1)
        elif action == 6:  # 提高模糊档位
            i_blur = min(2, i_blur + 1)
        elif action == 7:  # 降低噪声档位
            i_noise = max(0, i_noise - 1)
        elif action == 8:  # 提高噪声档位
            i_noise = min(2, i_noise + 1)
        elif action == 9:  # 无操作
            pass
        
        return [i_name, i_temp, i_fov, i_blur, i_noise]
    
    def reset(self, random_start: bool = True) -> np.ndarray:
        """重置环境"""
        if random_start:
            self.state = [
                self.rng.randint(0, 1),  # 传感器类型
                self.rng.randint(0, 2),  # 温度档位
                self.rng.randint(0, 2),  # 分辨率档位
                self.rng.randint(0, 2),  # 模糊档位
                self.rng.randint(0, 2),  # 噪声档位
            ]
        else:
            self.state = [0, 0, 0, 0, 0]  # 固定起点
        
        self.step_cnt = 0
        self.recent_gains = deque(maxlen=self.patience)
        self.trajectory = [self.state.copy()]  # 记录轨迹
        
        obs = self._encode_obs(self.state)
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        assert 0 <= action < self.n_actions(), f"Invalid action: {action}"
        
        old_f1 = self._f1(self.state)
        new_state = self._apply_action(self.state, action)
        new_f1 = self._f1(new_state)
        
        # 计算增量奖励
        reward = new_f1 - old_f1
        
        # 更新状态
        self.state = new_state
        self.step_cnt += 1
        self.recent_gains.append(abs(reward))
        self.trajectory.append(self.state.copy())
        
        # 判断终止条件
        done = False
        termination_reason = ""
        
        # 平台期检测：最近若干步增益均很小
        if len(self.recent_gains) == self.patience:
            if max(self.recent_gains) < self.epsilon_gain:
                done = True
                termination_reason = "plateau"
        
        # 步数限制
        if self.step_cnt >= self.max_steps:
            done = True
            termination_reason = "max_steps"
        
        obs = self._encode_obs(self.state)
        info = {
            "f1": new_f1,
            "f1_gain": reward,
            "action_name": self.action_names[action],
            "termination_reason": termination_reason,
            "steps": self.step_cnt
        }
        
        return obs, reward, done, info
    
    def current_combo_tuple(self) -> Tuple[int,int,int,int,int]:
        """获取当前状态的参数组合元组"""
        return tuple(self.state)
    
    def decode_current_combo(self) -> Dict:
        """解码当前状态为实际参数"""
        return self.param_space.decode_combo(self.current_combo_tuple())
    
    def get_trajectory_summary(self) -> Dict:
        """获取轨迹摘要信息"""
        f1_scores = [self._f1(state) for state in self.trajectory]
        
        return {
            "trajectory_length": len(self.trajectory),
            "start_f1": f1_scores[0] if f1_scores else 0.0,
            "end_f1": f1_scores[-1] if f1_scores else 0.0,
            "max_f1": max(f1_scores) if f1_scores else 0.0,
            "total_improvement": f1_scores[-1] - f1_scores[0] if len(f1_scores) >= 2 else 0.0,
            "trajectory": self.trajectory.copy()
        }


def generate_synthetic_f1(combo_tuple: Tuple[int,int,int,int,int], seed: int = 42) -> float:
    """
    根据参数组合生成合成F1值
    
    趋势设定：
    - LWIR (i_name=1) 稍好于 MWIR
    - 中温 (i_temp=1) 最优，过低/过高略差
    - 中分辨率 (i_fov=1) > 高分辨率 (i_fov=2) > 低分辨率 (i_fov=0)
    - 模糊/噪声越高，F1 越低
    - 加一点高斯噪声模拟随机性
    """
    i_name, i_temp, i_fov, i_blur, i_noise = combo_tuple
    
    # 设置随机种子以确保可重现性
    np.random.seed(seed + hash(combo_tuple) % 10000)
    
    # 基础分数
    base_score = 0.6
    
    # 传感器类型影响 (LWIR稍好)
    if i_name == 1:  # LWIR
        base_score += 0.08
    else:  # MWIR
        base_score += 0.03
    
    # 温度档位影响 (中温最优)
    temp_bonus = [0.02, 0.12, 0.06]  # [低温, 中温, 高温]
    base_score += temp_bonus[i_temp]
    
    # 分辨率影响 (中分辨率 > 高分辨率 > 低分辨率)
    fov_bonus = [0.04, 0.15, 0.10]  # [低分辨率, 中分辨率, 高分辨率]
    base_score += fov_bonus[i_fov]
    
    # 模糊程度影响 (越高越差)
    blur_penalty = [0.0, -0.06, -0.12]  # [无模糊, 中等模糊, 高模糊]
    base_score += blur_penalty[i_blur]
    
    # 噪声程度影响 (越高越差)
    noise_penalty = [0.0, -0.05, -0.10]  # [无噪声, 中等噪声, 高噪声]
    base_score += noise_penalty[i_noise]
    
    # 参数间的交互效应
    # LWIR + 中温 + 中分辨率的组合有额外奖励
    if i_name == 1 and i_temp == 1 and i_fov == 1:
        base_score += 0.05
    
    # 高模糊 + 高噪声的惩罚
    if i_blur == 2 and i_noise == 2:
        base_score -= 0.08
    
    # 添加高斯噪声
    noise_std = 0.03
    gaussian_noise = np.random.normal(0, noise_std)
    base_score += gaussian_noise
    
    # 确保F1值在合理范围内
    f1_score = np.clip(base_score, 0.1, 0.95)
    
    return float(f1_score)


def create_synthetic_cache(seed: int = 42, save_path: str = None) -> Dict[Tuple[int,int,int,int,int], float]:
    """
    创建合成的F1缓存数据集
    
    Args:
        seed: 随机种子
        save_path: 保存路径，如果为None则不保存
    
    Returns:
        包含所有参数组合F1分数的字典
    """
    print("[Synthetic] 开始生成合成F1数据集...")
    
    param_space = ParameterSpace()
    synthetic_cache = {}
    
    for combo in param_space.all_combos:
        f1_score = generate_synthetic_f1(combo, seed)
        synthetic_cache[combo] = f1_score
    
    # 统计信息
    f1_values = list(synthetic_cache.values())
    print(f"[Synthetic] 生成了 {len(synthetic_cache)} 个参数组合")
    print(f"[Synthetic] F1分数范围: {min(f1_values):.4f} - {max(f1_values):.4f}")
    print(f"[Synthetic] 平均F1分数: {np.mean(f1_values):.4f}")
    print(f"[Synthetic] F1分数标准差: {np.std(f1_values):.4f}")
    
    # 找出最优组合
    best_combo, best_f1 = max(synthetic_cache.items(), key=lambda x: x[1])
    param_space_instance = ParameterSpace()
    best_params = param_space_instance.decode_combo(best_combo)
    
    print(f"[Synthetic] 最优组合 F1={best_f1:.4f}:")
    for key, value in best_params.items():
        print(f"[Synthetic]   {key}: {value}")
    
    # 保存缓存
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(synthetic_cache, f)
        print(f"[Synthetic] 合成缓存已保存到: {save_path}")
    
    return synthetic_cache


def analyze_synthetic_trends(cache: Dict[Tuple[int,int,int,int,int], float]):
    """分析合成数据的趋势"""
    print("\n[Analysis] 合成数据趋势分析:")
    
    # 按传感器类型分组
    mwir_scores = [f1 for (i_name, _, _, _, _), f1 in cache.items() if i_name == 0]
    lwir_scores = [f1 for (i_name, _, _, _, _), f1 in cache.items() if i_name == 1]
    
    print(f"[Analysis] MWIR平均F1: {np.mean(mwir_scores):.4f}")
    print(f"[Analysis] LWIR平均F1: {np.mean(lwir_scores):.4f}")
    
    # 按温度档位分组
    temp_scores = {}
    for i_temp in range(3):
        scores = [f1 for (_, t, _, _, _), f1 in cache.items() if t == i_temp]
        temp_names = ["低温", "中温", "高温"]
        temp_scores[i_temp] = np.mean(scores)
        print(f"[Analysis] {temp_names[i_temp]}平均F1: {np.mean(scores):.4f}")
    
    # 按分辨率档位分组
    fov_scores = {}
    for i_fov in range(3):
        scores = [f1 for (_, _, f, _, _), f1 in cache.items() if f == i_fov]
        fov_names = ["低分辨率", "中分辨率", "高分辨率"]
        fov_scores[i_fov] = np.mean(scores)
        print(f"[Analysis] {fov_names[i_fov]}平均F1: {np.mean(scores):.4f}")
    
    # 按模糊/噪声程度分组
    for param_idx, param_name in [(3, "模糊"), (4, "噪声")]:
        param_names = ["无", "中等", "高"]
        for level in range(3):
            if param_idx == 3:
                scores = [f1 for (_, _, _, b, _), f1 in cache.items() if b == level]
            else:
                scores = [f1 for (_, _, _, _, n), f1 in cache.items() if n == level]
            print(f"[Analysis] {param_name}{param_names[level]}平均F1: {np.mean(scores):.4f}")


def exhaustive_search(combo_f1: Dict[Tuple[int,int,int,int,int], float]) -> Tuple[Tuple[int,int,int,int,int], float]:
    """从缓存中暴力搜索全局最优解"""
    #直接在 combo_f1 字典里取全局最大 F1（162 种组合，瞬间就找完）。
    #用来校验 DQN 学出来的答案是否接近/等于全局最优。直接在 combo_f1 字典里取全局最大 F1（162 种组合，瞬间就找完）。
    best = max(combo_f1.items(), key=lambda kv: kv[1])
    return best


if __name__ == "__main__":
    # 测试环境模块
    print("测试环境模块...")
    
    # 创建合成数据集
    print("\n=== 生成合成F1数据集 ===")
    synthetic_cache = create_synthetic_cache(seed=42, save_path='synthetic_combo_f1_cache.pkl')
    
    # 分析合成数据趋势
    analyze_synthetic_trends(synthetic_cache)
    
    # 测试暴力搜索
    print("\n=== 测试暴力搜索 ===")
    best_combo, best_f1 = exhaustive_search(synthetic_cache)
    param_space = ParameterSpace()
    best_params = param_space.decode_combo(best_combo)
    print(f"全局最优F1: {best_f1:.4f}")
    print("全局最优参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 创建环境并测试
    print("\n=== 测试DQN环境 ===")
    env = ParamOptEnv(combo_f1=synthetic_cache, seed=42)
    
    # 测试环境交互
    obs = env.reset()
    print(f"初始观测维度: {obs.shape}")
    print(f"初始状态: {env.decode_current_combo()}")
    print(f"初始F1: {env._f1(env.state):.4f}")
    
    # 执行几步测试
    for step in range(10):
        action = random.randint(0, env.n_actions() - 1)
        obs, reward, done, info = env.step(action)
        print(f"步骤 {step+1}: 动作={info['action_name']}, 奖励={reward:+.4f}, F1={info['f1']:.4f}")
        if done:
            print(f"环境终止: {info['termination_reason']}")
            break
    
    # 输出轨迹摘要
    summary = env.get_trajectory_summary()
    print(f"\n轨迹摘要:")
    print(f"轨迹长度: {summary['trajectory_length']}")
    print(f"起始F1: {summary['start_f1']:.4f}")
    print(f"结束F1: {summary['end_f1']:.4f}")
    print(f"最高F1: {summary['max_f1']:.4f}")
    print(f"总体改进: {summary['total_improvement']:.4f}")
    
    # 测试多个随机起点
    print(f"\n=== 测试多个随机起点 ===")
    improvements = []
    for trial in range(10):
        env.reset(random_start=True)
        start_f1 = env._f1(env.state)
        
        # 执行一些随机动作
        for _ in range(20):
            action = random.randint(0, env.n_actions() - 1)
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        final_f1 = info['f1']
        improvement = final_f1 - start_f1
        improvements.append(improvement)
        print(f"试验 {trial+1}: 起始F1={start_f1:.4f}, 结束F1={final_f1:.4f}, 改进={improvement:+.4f}")
    
    print(f"\n随机试验统计:")
    print(f"平均改进: {np.mean(improvements):+.4f}")
    print(f"改进标准差: {np.std(improvements):.4f}")
    print(f"正向改进率: {sum(1 for x in improvements if x > 0) / len(improvements):.2%}")
    
    print("\n环境模块测试完成！")
