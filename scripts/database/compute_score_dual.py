#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_score_dual.py
---------------------
针对两类目标（前 972 个 ConfigID 为 manbo，后 972 个为 panmao）的综合评分计算：
- 每 6 个 ConfigID 表示同一“图像参数组合”在 6 个不同场景下的结果；
- 每个参数组合在两个目标各有 6 个场景，共 12 个 ConfigID。

综合评分（score_all12）= 该参数组合在两类目标的 12 个场景的 avg_f1 简单平均（仅对现有数据求均值）。
同时输出单目标评分：score_manbo、score_panmao。

结果用于挑选“当前数据下综合评分最佳”的图像参数组合。
代表 ConfigID：按用户约定，取每个目标组内的第二个（即 group-of-6 的第 2 个）。

用法：
    python compute_score_dual.py --host 127.0.0.1 --port 3306 --user root --password 123456 --database config_db
"""

import argparse
import csv
import sys
from typing import List, Dict, Any, Optional

import mysql.connector


TOTAL = 1944
HALF  = 972   # 每个目标 972 个 ConfigID
SCENES_PER_TARGET = 6
COMBOS = HALF // SCENES_PER_TARGET  # 162


def target_of_configid(cfg: int) -> str:
    """根据用户给定的布局判定目标：前 972 为 manbo, 后 972 为 panmao."""
    return "manbo" if cfg <= HALF else "panmao"


def combo_index_of(cfg: int) -> int:
    """
    将 ConfigID 映射到参数组合索引（0..161）：
    - 先对 972 取模，落到当前目标的 0..971 范围；
    - 再除以 6（向下取整），得到 0..161。
    """
    within = (cfg - 1) % HALF  # 0..971
    return within // SCENES_PER_TARGET  # 0..161


def rep_configid_of(combo_idx: int, target: str) -> int:
    """
    代表 ConfigID：每目标内的该 combo 的第 2 个。
    目标起始偏移：manbo 从 1 开始，panmao 从 973 开始。
    """
    base = 1 if target == "manbo" else (HALF + 1)
    start_cfg = base + combo_idx * SCENES_PER_TARGET  # 该 combo 的第一个 ConfigID
    return start_cfg + 1  # 组内第 2 个


def fetch_per_configid_avg(conn) -> List[Dict[str, Any]]:
    """
    联结 images + evalution，聚合到 ConfigID 的平均 F1 与样本数。
    """
    sql = """
    SELECT
        i.ConfigID AS ConfigID,
        AVG(e.f1)  AS avg_f1,
        COUNT(*)   AS n_images
    FROM images AS i
    INNER JOIN evalution AS e ON e.image_id = i.image_id
    GROUP BY i.ConfigID
    ORDER BY i.ConfigID
    """
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    # 增添目标与参数组合索引
    for row in rows:
        cfg = int(row["ConfigID"])
        row["target"] = target_of_configid(cfg)
        row["combo_index"] = combo_index_of(cfg)
    return rows


def fetch_param_slice_for_config(conn, config_id: int) -> Optional[Dict[str, Any]]:
    """
    从 param_sets 读取 name → percentNoise 的图像参数切片。
    """
    sql = """
    SELECT
        name, maxTemperature, minTemperature, hFOVPixels, vFOVPixels,
        hFOVDeg, vFOVDeg, percentBlur, percentNoise
    FROM param_sets
    WHERE ConfigID = %s
    """
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, (config_id,))
    row = cur.fetchone()
    cur.close()
    return row


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_combo_scores(per_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 ConfigID 粒度的 avg_f1 聚合为参数组合粒度：
    - score_manbo: 该组合在 manbo 目标的 6 个场景平均（存在多少算多少）
    - score_panmao: 同上
    - score_all12: 两目标合并 12 个场景的平均（存在多少算多少）
    """
    # 初始化容器
    acc = [{
        "sum_manbo": 0.0, "cnt_manbo": 0,
        "sum_panmao": 0.0, "cnt_panmao": 0
    } for _ in range(COMBOS)]

    for row in per_cfg:
        idx = int(row["combo_index"])
        tgt = row["target"]
        avg = row["avg_f1"]
        if avg is None:
            continue
        avg = float(avg)
        if tgt == "manbo":
            acc[idx]["sum_manbo"] += avg
            acc[idx]["cnt_manbo"] += 1
        else:
            acc[idx]["sum_panmao"] += avg
            acc[idx]["cnt_panmao"] += 1

    out: List[Dict[str, Any]] = []
    for idx in range(COMBOS):
        s_m, c_m = acc[idx]["sum_manbo"], acc[idx]["cnt_manbo"]
        s_p, c_p = acc[idx]["sum_panmao"], acc[idx]["cnt_panmao"]
        score_m = (s_m / c_m) if c_m > 0 else None
        score_p = (s_p / c_p) if c_p > 0 else None

        # 合并 12 场景
        s_all = (s_m + s_p)
        c_all = (c_m + c_p)
        score_all = (s_all / c_all) if c_all > 0 else None

        out.append({
            "combo_index": idx,
            "score_all12": score_all,
            "n_scenes_all12": c_all,
            "score_manbo": score_m,
            "n_scenes_manbo": c_m,
            "score_panmao": score_p,
            "n_scenes_panmao": c_p,
            # 代表 ConfigID（每目标取组内第 2 个）
            "rep_cfg_manbo": rep_configid_of(idx, "manbo"),
            "rep_cfg_panmao": rep_configid_of(idx, "panmao"),
        })

    # 根据 score_all12 降序排序（None 排在最后）
    out.sort(key=lambda d: (-1 if d["score_all12"] is None else 0, -(d["score_all12"] or 0.0)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="asd515359")
    parser.add_argument("--database", default="config_db")
    args = parser.parse_args()

    try:
        conn = mysql.connector.connect(
            host=args.host, port=args.port,
            user=args.user, password=args.password,
            database=args.database, charset="utf8mb4"
        )
    except Exception as e:
        print("数据库连接失败：", e, file=sys.stderr)
        sys.exit(2)

    try:
        # 1) 每个 ConfigID 的平均 F1 + 标注目标 & 参数组合索引
        per_cfg = fetch_per_configid_avg(conn)
        save_csv("per_configid_avg.csv", per_cfg)
        print("已保存 per_configid_avg.csv")

        # 2) 计算每个参数组合跨两目标的综合评分
        combos = compute_combo_scores(per_cfg)
        save_csv("per_combo_scores.csv", combos)
        print("已保存 per_combo_scores.csv")

        if combos:
            best = next((c for c in combos if c["score_all12"] is not None), None)
            if best is None:
                print("未找到可计算评分的组合（可能 evalution 尚无数据）。")
                sys.exit(0)

            idx = best["combo_index"]
            rep_m = int(best["rep_cfg_manbo"])
            rep_p = int(best["rep_cfg_panmao"])

            # 读代表 ConfigID 的图像参数切片（任选其一作为图像参数代表，这里打印 manbo 代表）
            param_slice = fetch_param_slice_for_config(conn, rep_m)

            print("\n===== 当前数据下评分最优的图像参数组合 =====")
            print(f"参数组合索引（0-based）：{idx}  （共 {COMBOS} 个）")
            print(f"综合评分 score_all12：{best['score_all12']:.6f} （参与场景：{best['n_scenes_all12']} / 12）")
            print(f"manbo 子评分：{best['score_manbo'] if best['score_manbo'] is not None else 'NA'} "
                  f"（场景数：{best['n_scenes_manbo']} / 6）")
            print(f"panmao 子评分：{best['score_panmao'] if best['score_panmao'] is not None else 'NA'} "
                  f"（场景数：{best['n_scenes_panmao']} / 6）")
            print(f"代表 ConfigID（manbo）：{rep_m}")
            print(f"代表 ConfigID（panmao）：{rep_p}")
            if param_slice:
                print("图像参数切片（name → percentNoise，对应 manbo 代表 ConfigID）：")
                for k, v in param_slice.items():
                    print(f"  - {k}: {v}")
            else:
                print("未能在 param_sets 中读取到代表 ConfigID（manbo）的图像参数字段。")

            # 另外可打印该组合下所有 ConfigID 的 avg_f1（用于 sanity check）
            print("\n该组合下的 ConfigID 平均 F1 概览：")
            for row in per_cfg:
                if int(row["combo_index"]) == idx:
                    print(f"  [{row['target']}] ConfigID={row['ConfigID']}: "
                          f"avg_f1={row['avg_f1']:.6f} (n_images={row['n_images']})")

        else:
            print("没有可用的组合数据。")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
