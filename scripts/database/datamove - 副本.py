# -*- coding: utf-8 -*-
"""
批量汇总并导入 param_sets（不建表版）
- 方式A：一次性 INSERT ... SELECT（单事务，推荐）
- 方式B：按 ConfigID 分批 INSERT ... SELECT（超大数据可用），避免长事务和大锁
"""

import sys
import mysql.connector as mc

# ======= 必填：连接信息 =======
DB_Config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'asd515359',
    'database': 'config_db',
    'port' : 3306
}
# 是否按 ConfigID 分批执行（True=分批；False=一次性）
BATCH_BY_CONFIGID = True
BATCH_SIZE = 500   # 每批多少个 ConfigID（仅在 BATCH_BY_CONFIGID=True 时生效）

# 若 param_sets.ConfigID 有唯一键/主键，是否使用 upsert（ON DUPLICATE KEY UPDATE）
USE_UPSERT = True


# ======= 公共 SQL 片段：SELECT 聚合部分 =======
# 说明：
# - 通过 CASE WHEN 做干扰类平均坐标
# - 通过 MAX(CASE ...) 做实体存在标志
# - GROUP BY 覆盖所有非聚合列，兼容 ONLY_FULL_GROUP_BY
SELECT_CORE = r"""
  SELECT
    s.ConfigID, s.name,
    s.hFOVPixels, s.vFOVPixels, s.hFOVDeg, s.vFOVDeg, s.maxTemperature, s.minTemperature,
    s.senSim_percentBlur, s.senSim_percentNoise,

--     AVG(CASE WHEN INSTR(e.entityName,'hajimi')>0 OR INSTR(e.entityName,'panbaobao')>0
--              THEN e.worldXCoord END) AS meanInterXCoord,
--     AVG(CASE WHEN INSTR(e.entityName,'hajimi')>0 OR INSTR(e.entityName,'panbaobao')>0
--              THEN e.worldYCoord END) AS meanInterYCoord,
--     AVG(CASE WHEN INSTR(e.entityName,'hajimi')>0 OR INSTR(e.entityName,'panbaobao')>0
--              THEN e.worldZCoord END) AS meanInterZCoord,

    MAX(INSTR(e.entityName,'manbo')>0)      AS ismanbo,
    MAX(INSTR(e.entityName,'panmao')>0)     AS ispanmao,
    MAX(INSTR(e.entityName,'hajimi')>0) AS ishajimi,
    MAX(INSTR(e.entityName,'panbaobao')>0)      AS ispanbaobao,
    env.`rainRate`,
    env.hazeModel,
    env.`time`,
    env.`date`
  FROM sensors AS s
  LEFT JOIN entities AS e
    ON e.ConfigID = s.ConfigID
  LEFT JOIN `environment-manager` AS env
    ON env.ConfigID = s.ConfigID
"""

GROUP_BY = r"""
  GROUP BY
    s.ConfigID, s.name,
    s.hFOVPixels, s.vFOVPixels, s.hFOVDeg, s.vFOVDeg,s.maxTemperature, s.minTemperature,
    s.senSim_percentBlur, s.senSim_percentNoise,
    env.hazeModel, env.`time`,env.`date`
"""

# 目标列清单（顺序要与 SELECT 字段顺序完全一致）
TARGET_COLS = """
  (ConfigID, name,
   hFOVPixels, vFOVPixels, hFOVDeg, vFOVDeg,maxTemperature, minTemperature,
   percentBlur, percentNoise,
   # meanInterXCoord, meanInterYCoord, meanInterZCoord,
   ismanbo, ispanmao, ishajimi, ispanbaobao,
   rainRate,hazeModel, `time`,`date`)
"""

# upsert 子句（当 param_sets.ConfigID 唯一时可用）
UPSERT_CLAUSE = r"""
  ON DUPLICATE KEY UPDATE
    name=VALUES(name),
    hFOVPixels=VALUES(hFOVPixels),
    vFOVPixels=VALUES(vFOVPixels),
    hFOVDeg=VALUES(hFOVDeg),
    vFOVDeg=VALUES(vFOVDeg),
    maxTemperature=VALUES(maxTemperature),
    minTemperature=VALUES(minTemperature),
    percentBlur=VALUES(percentBlur),
    percentNoise=VALUES(percentNoise),
    # meanInterXCoord=VALUES(meanInterXCoord),
    # meanInterYCoord=VALUES(meanInterYCoord),
    # meanInterZCoord=VALUES(meanInterZCoord),
    ismanbo=VALUES(ismanbo),
    ispanmao=VALUES(ispanmao),
    ishajimi=VALUES(ishajimi),
    ispanbaobao=VALUES(ispanbaobao),
    rainRate=VALUES(rainRate),
    hazeModel=VALUES(hazeModel),
    `time`=VALUES(`time`),
    `date`=VALUES(`date`)
"""


def insert_full(conn):
    """方式A：一次性 INSERT ... SELECT（单事务）"""
    cur = conn.cursor()
    try:
        sql = f"""
          INSERT INTO param_sets {TARGET_COLS}
          {SELECT_CORE}
          {GROUP_BY}
        """
        if USE_UPSERT:
            sql += UPSERT_CLAUSE

        cur.execute(sql)
        conn.commit()
        print("DONE(single shot). rows affected (approx):", cur.rowcount)
    finally:
        cur.close()


def get_configid_range(conn):
    """获取 ConfigID 的最小/最大值"""
    cur = conn.cursor()
    try:
        cur.execute("SELECT MIN(ConfigID), MAX(ConfigID) FROM sensors")
        r = cur.fetchone()
        return r if r else (None, None)
    finally:
        cur.close()


def insert_by_batches(conn, batch_size=BATCH_SIZE):
    """方式B：按 ConfigID 连续区间分批 INSERT ... SELECT，避免长事务和大锁"""
    cur = conn.cursor()
    try:
        min_id, max_id = get_configid_range(conn)
        if min_id is None or max_id is None:
            print("No data found in sensors.")
            return

        total_rows = 0
        start = min_id
        while start <= max_id:
            end = start + batch_size - 1

            sql = f"""
              INSERT INTO param_sets {TARGET_COLS}
              {SELECT_CORE}
              WHERE s.ConfigID BETWEEN %s AND %s
              {GROUP_BY}
            """
            if USE_UPSERT:
                sql += UPSERT_CLAUSE

            cur.execute(sql, (start, end))
            conn.commit()
            total_rows += cur.rowcount
            print(f"batch [{start}, {end}] committed, rows≈{cur.rowcount}")

            start = end + 1

        print("DONE(batched). total rows affected (approx):", total_rows)
    finally:
        cur.close()


def main():
    conn = mc.connect(**DB_Config)
    try:
        if BATCH_BY_CONFIGID:
            insert_by_batches(conn, BATCH_SIZE)
        else:
            insert_full(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except mc.Error as e:
        print("MySQL Error:", e, file=sys.stderr)
        sys.exit(1)
