#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从本地 config.ini 读取数据库连接，向 evalution(image_id, f1, model) 写入数据：
- image_id = 1..MAX(ConfigID)（来自 image_id 表）
- f1 以“0/1 长段 + 分散特殊值(1/3, 1/2, 2/3) 单点”生成
- 严格保证：特殊值不连续出现（可调 FRAC_MIN_GAP 控制最小间隔）
- 分块 executemany 流式插入，适合百万/千万级数据

注意：表名用 evalution（按你给的拼写）；如实际不同请改 TABLE_EVALUTION。
"""

import configparser
import math
import random
from typing import Iterator, List, Tuple
from pathlib import Path
import mysql.connector


# ======================= 可直接修改的参数 =======================
filepath = Path(__file__).resolve().parent
CONFIG_INI_PATH = filepath / 'config.ini'      # 配置文件路径

# 数据表名（与你实际库一致即可）
TABLE_image_id = "images"
TABLE_EVALUTION = "evalution"

# 固定写入的 model 值
MODEL_VALUE = 0

# 分块大小：每批插入行数
CHUNK_SIZE = 5000

# 随机种子：固定后便于复现实验
SEED = random.randint(0,2025)

# 0/1 段（binary run）的长度范围（建议 1000~20000，根据库性能调整）
MIN_RUN = 480
MAX_RUN = 960

# 抽到“尝试特殊值”的概率（其余则走 0/1 段）
# 提示：如果你想“更多 0/1 段、少量特殊值”，提高 P_BINARY（如 0.7~0.9）
P_BINARY = 0.1

# 特殊值集合及权重（1/2 略多一点点）
FRAC_CHOICES = (1.0/3.0, 0.5, 2.0/3.0)
FRAC_WEIGHTS = (0.33, 0.33, 0.34)

# 特殊值的最小间隔：两次出现特殊值之间至少隔多少个样本（默认 1）
# 例如 1 表示 “特殊值、(0/1)、特殊值、(0/1)、特殊值...” 允许；0 则允许相邻（不建议）
FRAC_MIN_GAP = 1

# f1 保留小数位（为了写库更规整；如表字段是 DECIMAL(4,3) 推荐 3）
F1_DECIMALS = 3

# 是否打印 run 信息与进度
VERBOSE = False
# ============================================================================


def load_db_configs(ini_path: str):
    """读取 config.ini，返回一个或多个数据库目标的配置字典列表。"""
    cfg = configparser.ConfigParser()
    read_files = cfg.read(ini_path, encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"未找到配置文件：{ini_path}")

    sections = []
    for sec in cfg.sections():
        d = {k.lower(): v.strip() for k, v in cfg[sec].items()}
        for key in ("host", "port", "user", "password", "database"):
            if key not in d:
                raise ValueError(f"[{sec}] 缺少必要配置项：{key}")
        d["port"] = int(d["port"])
        d["__name__"] = sec
        sections.append(d)
    if not sections:
        raise ValueError("配置文件中没有可用的 section。")
    return sections


def get_max_config_id(conn) -> int:
    """读取 image_id 表中的 MAX(ConfigID)。"""
    sql = f"SELECT MAX(image_id) FROM {TABLE_image_id}"
    cur = conn.cursor()
    try:
        cur.execute(sql)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        cur.close()


def _choose_fraction() -> float:
    """按权重选择一个特殊值，并做四舍五入（F1_DECIMALS）。"""
    r = random.random()
    acc = 0.0
    for v, w in zip(FRAC_CHOICES, FRAC_WEIGHTS):
        acc += w
        if r <= acc:
            return round(v, F1_DECIMALS)
    return round(FRAC_CHOICES[-1], F1_DECIMALS)


def _choose_binary_value() -> float:
    """随机选择 0 或 1，并做四舍五入（与 DECIMAL 字段对齐）。"""
    return round(1.0 if random.random() < 0.55 else 0.0, F1_DECIMALS)


def gen_records_stream(total_ids: int) -> Iterator[Tuple[int, float, int]]:
    """
    生成 (image_id, f1, model) 流式记录：
      - 特殊值强制为“单点”（run 长度=1）
      - 保证特殊值之间至少间隔 FRAC_MIN_GAP
      - 0/1 段仍然是大段 run（MIN_RUN..MAX_RUN）
    """
    current_id = 1
    last_frac_pos = -10**12  # 最近一次特殊值出现的位置；初始化为极小

    while current_id <= total_ids:
        # 是否尝试放一个“特殊值单点”
        try_fraction = (random.random() >= P_BINARY)  # 与 P_BINARY 互补

        if try_fraction and (current_id - last_frac_pos) > FRAC_MIN_GAP:
            # 放一个“非连续”的特殊值单点
            f1_val = _choose_fraction()
            yield (current_id, f1_val, MODEL_VALUE)
            last_frac_pos = current_id
            current_id += 1
            continue

        # 否则放一个 0/1 大段 run
        run_len = random.randint(MIN_RUN, MAX_RUN)
        # 注意剩余长度截断
        run_len = min(run_len, total_ids - current_id + 1)
        bin_val = _choose_binary_value()
        if VERBOSE:
            print(f"[RUN] id={current_id}..{current_id+run_len-1}, f1={bin_val}")
        for _ in range(run_len):
            yield (current_id, bin_val, MODEL_VALUE)
            current_id += 1
        # 这里不更新 last_frac_pos，因为这是 binary 段


def chunked(iterable: Iterator[Tuple[int, float, int]], chunk_size: int) -> Iterator[List[Tuple[int, float, int]]]:
    """将迭代器按 chunk_size 切块。"""
    batch: List[Tuple[int, float, int]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_one_database(conf: dict):
    """对单个数据库执行生成与插入流程。"""
    name = conf.get("__name__", "<unnamed>")
    print(f"\n====== 开始处理数据库 [{name}] {conf['user']}@{conf['host']}:{conf['port']}/{conf['database']} ======")

    conn = mysql.connector.connect(
        host=conf["host"],
        port=conf["port"],
        user=conf["user"],
        password=conf["password"],
        database=conf["database"],
        autocommit=False,
    )

    try:
        total_ids = get_max_config_id(conn)
        if total_ids <= 0:
            print("image_id 表中未找到有效的 ConfigID（MAX(ConfigID) 为 NULL 或 0）。跳过该库。")
            return

        print(f"MAX(ConfigID) = {total_ids}，将按 image_id=1..{total_ids} 写入 {TABLE_EVALUTION}。")
        print(f"chunk={CHUNK_SIZE}, binary run=[{MIN_RUN}, {MAX_RUN}], P_BINARY={P_BINARY}, FRAC_MIN_GAP={FRAC_MIN_GAP}")

        sql = f"INSERT INTO {TABLE_EVALUTION} (image_id, f1, model) VALUES (%s, %s, %s)"
        cursor = conn.cursor()

        total_batches = math.ceil(total_ids / CHUNK_SIZE)
        done_rows = 0
        batch_index = 0

        for batch in chunked(gen_records_stream(total_ids), CHUNK_SIZE):
            batch_index += 1
            try:
                cursor.executemany(sql, batch)
                conn.commit()
            except mysql.connector.Error as e:
                conn.rollback()
                print(f"[ERROR] 第 {batch_index}/{total_batches} 批插入失败，已回滚。错误：{e}")
                raise

            done_rows += len(batch)
            if VERBOSE or batch_index % 10 == 0 or batch_index == total_batches:
                print(f"进度：{batch_index}/{total_batches} 批，累计写入 {done_rows}/{total_ids} 行")

        cursor.close()
        print(f"数据库 [{name}] 全部插入完成。")
    finally:
        conn.close()


def main():
    random.seed(SEED)
    db_confs = load_db_configs(CONFIG_INI_PATH)
    for conf in db_confs:
        process_one_database(conf)


if __name__ == "__main__":
    main()
