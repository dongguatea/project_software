# csv2mysql.py
# --------------------------------------------------
# 批量读取 csv_data/*.csv  →  写入同名 MySQL 表
# --------------------------------------------------
import glob, os, sys
import pandas as pd
import mysql.connector
from pathlib import Path

# ---------- 数据库连接信息 ----------
DB_CONFIG = dict(
    host="localhost",
    port=3306,
    user="root",
    password="asd515359",
    database="config_db",
    charset ="utf8",
    autocommit=True           # 自动提交，方便脚本反复执行
)
filepath = Path(__file__).resolve()
csvpath = filepath.parent / 'csv'

CSV_DIR= filepath.parent / 'generate'
CHUNK= 1000               # 大文件分块行数

# ---------- 建立连接 ----------
try:
    conn = mysql.connector.connect(**DB_CONFIG)
except mysql.connector.Error as e:
    sys.exit(f"无法连接数据库: {e}")
cursor = conn.cursor(dictionary=True)

def table_columns(table):
    """返回表字段集合（小写）"""
    cursor.execute(f"SHOW COLUMNS FROM `{table}`;")
    return {row["Field"].lower() for row in cursor.fetchall()}

# ---------- 遍历所有 CSV ----------
for csv_path in glob.glob(os.path.join(CSV_DIR, "*.csv")):
    table = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        cols_db = table_columns(table)
    except mysql.connector.Error:
        print(f" 跳过 {csv_path}（表 `{table}` 不存在）")
        continue
    print(f"导入 {csv_path} → 表 `{table}`")

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK):
        if table == "scenario":
            if "Use_ConfigID" not in chunk.columns:
                chunk["Use_ConfigID"] = 0
            else:
                chunk["Use_ConfigID"].fillna(0, inplace=True)
        # 统一列名去空格
        chunk.rename(columns=lambda c: c.strip(), inplace=True)
        valid_cols = [c for c in chunk.columns if c.lower() in cols_db]
        if not valid_cols:
            print(f"该分块无有效列，跳过")
            continue

        # 生成 INSERT 语句
        cols_fmt = ", ".join(f"`{c}`" for c in valid_cols)
        placeholders = ", ".join(["%s"] * len(valid_cols))
        sql = f"INSERT INTO `{table}` ({cols_fmt}) VALUES ({placeholders})"
        data = [tuple(chunk[c].where(pd.notna(chunk[c]), None)) for c in valid_cols]
        # data = list(zip(*(chunk[c] for c in valid_cols)))  # 等价

        try:
            cursor.executemany(sql, list(zip(*data)))
        except mysql.connector.Error as e:
            print(f"插入失败: {e}")
            conn.rollback()
            continue
        print(f"  + {cursor.rowcount} 行")

print("CSV 全部导入完毕")
cursor.close()
conn.close()
