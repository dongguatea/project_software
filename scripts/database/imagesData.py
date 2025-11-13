# -*- coding: utf-8 -*-
import sys
import mysql.connector as mc

# ======= 基本配置（按需修改） =======
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "asd515359",
    "database": "config_db",
}

TABLE_PARAM_SET = "param_sets"     # ConfigID 来源
TABLE_ANGLES    = "angles"         # 列: R, theta, phi
TABLE_IMAGES    = "images"         # 目标: image_id, ConfigID, theta, phi, R, image_path

BASE_DIR = "F:/data_manager/pythonProject1/data/images"                # 生成路径前缀：images/{ConfigID}/{angle_idx:05d}.bmp

FETCH_BATCH_SIZE  = 50_000         # 每次从 SELECT 拉这么多
INSERT_BATCH_SIZE = 10_000         # 每次 executemany 插这么多
# ===================================

# 1) 建议存在的联合索引（包含 R；已存在会忽略）
SQL_CREATE_INDEX_IF_ABSENT = f"""
ALTER TABLE {TABLE_IMAGES}
ADD INDEX idx_images_cfg_theta_phi_r (ConfigID, theta, phi, R)
"""

# 2) 拉 angles，建立 (R,theta,phi) → angle_idx 映射（按固定顺序，从 0 开始）
SQL_LOAD_ANGLES = f"""
SELECT id,R, theta, phi
FROM {TABLE_ANGLES}
ORDER BY id
"""

# 3) 在服务器端挑“缺失组合”，把 R 也作为去重键
SQL_SELECT_MISSING = f"""
SELECT ps.ConfigID, a.R, a.theta, a.phi
FROM {TABLE_PARAM_SET} AS ps
CROSS JOIN {TABLE_ANGLES} AS a
LEFT JOIN {TABLE_IMAGES} AS i
  ON i.ConfigID = ps.ConfigID
 AND i.theta    = a.theta
 AND i.phi      = a.phi
 AND i.R        = a.R
WHERE i.image_id IS NULL
ORDER BY ps.ConfigID,a.id
"""

# 4) 分块 executemany 插入，直接带上 image_path
SQL_INSERT_IMAGES = f"""
INSERT INTO {TABLE_IMAGES} (ConfigID, theta, phi, R, image_path)
VALUES (%s, %s, %s, %s, %s)
"""

def iter_fetchmany(cur, size):
    """流式拉取结果，按块返回。"""
    while True:
        rows = cur.fetchmany(size)  # 注意：mysql-connector 用位置参数 size
        if not rows:
            break
        yield rows

def chunk_rows(rows, n):
    """把一大块再切成更小块用于 executemany。"""
    for i in range(0, len(rows), n):
        yield rows[i:i+n]

def main():
    conn_read  = mc.connect(**DB_CONFIG)
    conn_write = mc.connect(**DB_CONFIG)
    conn_write.autocommit = False

    cur_read = conn_read.cursor(buffered=False)   # 只负责 SELECT + fetchmany
    cur_exec = conn_write.cursor(prepared=True)   # 只负责 executemany 插入

    try:
        # A)（可选）确保有利于缺失判断/联表的索引（包含 R）
        try:
            cur_exec.execute(SQL_CREATE_INDEX_IF_ABSENT)
            conn_write.commit()
        except mc.Error as e:
            if getattr(e, "errno", None) not in (1061, 1060):  # Duplicate key/column
                raise

        # B) 读取 angles，建立 (R,theta,phi) -> angle_idx 映射（有序稳定）
        cur_tmp = conn_read.cursor()
        cur_tmp.execute(SQL_LOAD_ANGLES)
        angle_rows = cur_tmp.fetchall()  # [(R, theta, phi), ...]
        cur_tmp.close()

        # 从 0 开始编号
        angle_index = {(R, theta, phi): idx for idx, (id,R, theta, phi) in enumerate(angle_rows)}
        print(angle_index)
        if not angle_index:
            print("angles 表为空，无法生成路径。", file=sys.stderr)
            conn_write.rollback()
            return

        # C) 选择缺失组合（含 R），流式拉取
        cur_read.execute(SQL_SELECT_MISSING)


        total_inserted = 0
        for big_batch in iter_fetchmany(cur_read, FETCH_BATCH_SIZE):
            # big_batch: [(ConfigID, R, theta, phi), ...] —— 注意顺序
            prepared_rows = []
            for cfg, R, theta, phi in big_batch:  # 顺序与 SQL_SELECT_MISSING 对齐
                idx = angle_index.get((R, theta, phi))
                if idx is None:
                    # 角度组合在映射表里找不到（角度表与缺失集不一致），可选择 raise
                    # raise ValueError(f"angles 缺失 (R={R}, theta={theta}, phi={phi})")
                    continue

                image_path = f"{BASE_DIR}/{cfg}/{idx:05d}.bmp"
                prepared_rows.append((cfg, theta, phi, R, image_path))

            # 分块 executemany
            for small_batch in chunk_rows(prepared_rows, INSERT_BATCH_SIZE):
                cur_exec.executemany(SQL_INSERT_IMAGES, small_batch)
                conn_write.commit()
                total_inserted += len(small_batch)

        print(f"完成插入：{total_inserted} 行（已包含 image_path，避免后续 UPDATE）")

    except Exception as e:
        conn_write.rollback()
        print("❌ 出错，已回滚：", e, file=sys.stderr)
        raise
    finally:
        try:
            cur_read.close()
        except:
            pass
        try:
            cur_exec.close()
        except:
            pass
        try:
            conn_read.close()
        except:
            pass
        try:
            conn_write.close()
        except:
            pass

if __name__ == "__main__":
    main()
