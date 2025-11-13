#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_score.py
----------------
基于现有三表（param_sets, images, evalution），计算“图像参数组合”的综合评分系数，
并输出当前数据下最好的图像参数组合（以每 6 个 ConfigID 为一组，取组内第 2 个 ConfigID 作为代表）。

MySQL 5.7 兼容，无 CTE。

输出：
1) per_configid_avg.csv    —— 每个 ConfigID 的平均 F1（现有数据下能算出来的）
2) per_group_scores.csv    —— 每组（图像参数组合）综合评分（组内场景均值）、可用场景数、代表 ConfigID
3) 控制台打印：当前最优的图像参数组合（代表 ConfigID）及其参数（从 name 到 percentNoise），以及评分。

用法：
    python compute_score.py --host 127.0.0.1 --port 3306 --user root --password 123456 --database config_db
"""

import argparse
import csv
import sys
from typing import List, Dict, Any, Optional

import mysql.connector


def fetch_per_configid_avg(conn) -> List[Dict[str, Any]]:
    """
    计算每个 ConfigID 的平均 F1。
    注意：只对 evalution 中已有的 image_id 进行聚合；缺失的数据不会参与平均。
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
    return rows


def fetch_param_slice_for_config(conn, config_id: int) -> Optional[Dict[str, Any]]:
    """
    读取 param_sets 中“图像参数字段切片”（从 name 到 percentNoise）。
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


def group_id_of(config_id: int) -> int:
    """0-based 分组编号：每 6 个为一组"""
    return (config_id - 1) // 6


def rep_configid_of_group(group_id: int) -> int:
    """代表 ConfigID：每组取第 2 个（1-based 的第 2 个），即 0-based 组号 * 6 + 2"""
    return group_id * 6 + 2


def compute_group_scores(per_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    基于每个 ConfigID 的 avg_f1，汇总为每个“图像参数组合组”的综合评分：
    - group_mean_all：组内各场景 avg_f1 的简单平均（只平均已存在的场景）
    - n_scenes：本组参与平均的场景数（<=6）
    - rep_configid：代表 ConfigID（第 2 个）
    """
    # 汇总到 group_id
    groups: Dict[int, Dict[str, Any]] = {}
    for row in per_cfg:
        cfg = int(row["ConfigID"])
        avg_f1 = float(row["avg_f1"]) if row["avg_f1"] is not None else None
        gid = group_id_of(cfg)
        if gid not in groups:
            groups[gid] = {"sum": 0.0, "cnt": 0, "cfg_list": []}
        if avg_f1 is not None:
            groups[gid]["sum"] += avg_f1
            groups[gid]["cnt"] += 1
        groups[gid]["cfg_list"].append(cfg)

    # 生成输出列表
    out: List[Dict[str, Any]] = []
    for gid, acc in groups.items():
        cnt = acc["cnt"]
        group_mean_all = (acc["sum"] / cnt) if cnt > 0 else None
        rep_cfg = rep_configid_of_group(gid)
        out.append({
            "group_id": gid,
            "rep_configid": rep_cfg,
            "group_mean_all": group_mean_all,
            "n_scenes": cnt
        })
    # 按评分降序（空值放后）
    out.sort(key=lambda d: (-1 if d["group_mean_all"] is None else 0, -(d["group_mean_all"] or 0.0)))
    return out


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
        # 1) 每个 ConfigID 的平均 F1
        per_cfg = fetch_per_configid_avg(conn)
        save_csv("per_configid_avg.csv", per_cfg)
        print("已保存 per_configid_avg.csv")

        # 2) 计算组级综合评分（图像参数组合）
        groups = compute_group_scores(per_cfg)
        save_csv("per_group_scores.csv", groups)
        print("已保存 per_group_scores.csv")

        if groups:
            # 3) 选出评分最高的组（忽略 None）
            best = next((g for g in groups if g["group_mean_all"] is not None), None)
            if best is None:
                print("未找到可计算评分的组（可能 evalution 尚无数据）。")
                sys.exit(0)

            rep_cfg = int(best["rep_configid"])
            param_slice = fetch_param_slice_for_config(conn, rep_cfg)

            print("\n===== 当前数据下评分最优的图像参数组合 =====")
            print(f"代表 ConfigID：{rep_cfg}")
            print(f"组编号（0-based）：{best['group_id']}")
            print(f"综合评分（组内场景均值）：{best['group_mean_all']:.6f}")
            print(f"参与场景数：{best['n_scenes']} / 6")
            if param_slice:
                print("图像参数（name → percentNoise）：")
                for k, v in param_slice.items():
                    print(f"  - {k}: {v}")
            else:
                print("未能在 param_sets 中读取到该代表 ConfigID 的图像参数字段。")

            # 可选：也可顺带打印该组内每个 ConfigID 的 avg_f1
            gid = best["group_id"]
            cfg_of_group = [row for row in per_cfg if group_id_of(int(row["ConfigID"])) == gid]
            print("\n该组场景平均 F1：")
            for row in cfg_of_group:
                print(f"  ConfigID={row['ConfigID']}: avg_f1={row['avg_f1']:.6f} (n_images={row['n_images']})")

        else:
            print("没有可用的组数据。")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
