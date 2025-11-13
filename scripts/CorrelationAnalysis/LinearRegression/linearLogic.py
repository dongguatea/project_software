import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import errorcode

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import statsmodels.api as sm

# ========= 配置区 =========
DB_CFG = {
    "host":     os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PWD",  "asd515359"),
    "database": os.getenv("MYSQL_DB",   "config_db"),
    "charset":  "utf8mb4",
}
TABLE_EVAL = "evalution"
TARGET_COL = "f1"

TEST_SIZE = 0.2
RANDOM_STATE = 42
ROUND_DISCRETE = 3      # 连续离散量四舍五入到 3 位后再转字符串

EXPORT_ENGINEERED_CSV = False
ENGINEERED_CSV_PATH = "engineered_dataset_discrete_ohe.csv"
# ========================

# 建立数据库连接
def get_conn():
    try:
        return mysql.connector.connect(**DB_CFG)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("数据库账户或密码错误"); sys.exit(1)
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("数据库不存在"); sys.exit(1)
        else:
            print(err); sys.exit(1)
# 从数据库获取数据
def fetch_dataframe():
    conn = get_conn()
    try:
        cur = conn.cursor()
        sql = f"""
            SELECT
                ps.*,
                i.R       AS R,
                i.theta   AS theta,
                i.phi     AS phi,
                es.f1     AS f1
            FROM param_sets ps
            LEFT JOIN images i
                ON ps.ConfigID = i.ConfigID
            LEFT JOIN {TABLE_EVAL} es
                ON i.image_id = es.image_id;
        """
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        data = cur.fetchall()
        return pd.DataFrame(data, columns=cols)
    finally:
        conn.close()
def make_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) 删除 FOV 角度列（与像素严格线性相关）
    df = df.drop(columns=[c for c in ["hFOVDeg", "vFOVDeg"] if c in df.columns], errors="ignore")

    # 2) 分辨率类别
    if {"hFOVPixels", "vFOVPixels"}.issubset(df.columns):
        df["resolution_cls"] = (
            df["hFOVPixels"].astype("Int64").astype(str) + "x" +
            df["vFOVPixels"].astype("Int64").astype(str)
        )
    else:
        df["resolution_cls"] = np.nan

    # 3) 温差（离散）
    if {"maxTemperature","minTemperature"}.issubset(df.columns):
        df["temperature_range"] = df["maxTemperature"] - df["minTemperature"]
    else:
        df["temperature_range"] = np.nan

    # 4) Haze 条件化可视域/雨量（只在 hazeModel==0 时才有效）
    def _to_str_discrete(x): #传入列名，将列值四舍五入后离散化为字符串
        # 所有离散连续量统一四舍五入到 ROUND_DISCRETE 位再转字符串
        if pd.isna(x): return np.nan
        return str(np.round(float(x), ROUND_DISCRETE))

    # visibility_cond / rainrate_cond
    if "hazeModel" in df.columns:
        haze0 = (df["hazeModel"].astype("Int64") == 0)
    else:
        haze0 = pd.Series(False, index=df.index)

    if "visibility" in df.columns:
        df["visibility_cond"] = np.where(
            haze0, df["visibility"].map(_to_str_discrete), 0
        )
    else:
        df["visibility_cond"] = np.nan

    if "rainRate" in df.columns:
        df["rainrate_cond"] = np.where(
            haze0, df["rainRate"].map(_to_str_discrete), 0
        )
    else:
        df["rainrate_cond"] = np.nan

    # 5) 统一规范布尔列（仍按类别处理）
    for bcol in ["ismanbo","ispanmao","ishajimi","ispanbaobao"]:
        if bcol in df.columns and df[bcol].dtype == object:
            df[bcol] = df[bcol].str.lower().map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0})

    # 6) time → 字符串（如 '0:00','6:00','12:00','18:00'）
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    # 7) R/theta/phi 转字符串（离散档位）
    for c in ["R","theta","phi"]:
        if c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(0).astype("Int64").astype(str)
            elif pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype("Int64").astype(str)
            else:
                df[c] = df[c].astype(str)

    # 8) 其他需要离散化到字符串的连续离散量
    for c in ["percentBlur","percentNoise","temperature_range"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: str(np.round(float(x), ROUND_DISCRETE)) if pd.notna(x) else np.nan)

    # 9) hazeModel 也作为类别（0,3,9）
    if "hazeModel" in df.columns:
        df["hazeModel"] = df["hazeModel"].astype("Int64").astype(str)

    return df

def build_feature_target(df: pd.DataFrame):
    # 因变量限制在 [0,1]
    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到因变量列：{TARGET_COL}")
    df = df.dropna(subset=[TARGET_COL])
    df = df[(df[TARGET_COL] >= 0.0) & (df[TARGET_COL] <= 1.0)]
    y = df[TARGET_COL].astype(float)

    # 删除主键/泄露列
    df = df.drop(columns=[c for c in ["ConfigID","image_id"] if c in df.columns], errors="ignore")

    # 选择自变量（全部按“类别字符串”处理）
    candidate_features = [
        # param_sets
        "name","temperature_range","resolution_cls",
        "percentBlur","percentNoise","hazeModel",
        "visibility_cond","rainrate_cond",
        "time",
        "ismanbo","ispanmao","ishajimi","ispanbaobao",
        # images
        "R","theta","phi",
    ]
    feat_cols = [c for c in candidate_features if c in df.columns]
    X = df[feat_cols].copy()
    # 统一转字符串（保险）
    for c in feat_cols:
        X[c] = X[c].astype(str)

    return X, y, feat_cols

def fit_fractional_logit(X, y):
    # 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 众数填充 + 全量 One-Hot（drop='first' 避免虚拟变量陷阱）
    preproc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    Xtr = preproc.fit_transform(X_train)
    Xte = preproc.transform(X_test)

    # Fractional Logit（GLM Binomial with logit link）
    Xtr_const = sm.add_constant(Xtr, has_constant="add")
    glm = sm.GLM(y_train.values.astype(float), Xtr_const, family=sm.families.Binomial()).fit()

    # 概率预测与指标
    Xte_const = sm.add_constant(Xte, has_constant="add")
    y_prob = glm.predict(Xte_const)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    eps = 1e-15
    brier = float(np.mean((y_test.values - y_prob) ** 2))
    logloss = float(-np.mean(y_test.values * np.log(y_prob + eps) + (1 - y_test.values) * np.log(1 - y_prob + eps)))

    # McFadden 伪 R²（训练集）
    X0 = np.ones((len(y_train), 1))
    null_fit = sm.GLM(y_train.values.astype(float), X0, family=sm.families.Binomial()).fit()
    pseudo_r2 = 1.0 - (glm.llf / null_fit.llf)

    # 特征名
    oh = preproc.named_steps["onehot"]
    feature_names = ["const"] + oh.get_feature_names_out(X.columns).tolist()

    return {
        "preproc": preproc,
        "glm": glm,
        "feature_names": feature_names,
        "metrics": {"Brier": brier, "LogLoss": logloss, "McFaddenR2_train": float(pseudo_r2)},
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "y_prob": y_prob
    }

def export_engineered(preproc, X, y, path):
    design = preproc.transform(X)
    oh = preproc.named_steps["onehot"]
    names = ["const"] + oh.get_feature_names_out(X.columns).tolist()
    out = pd.DataFrame(np.column_stack([np.ones((design.shape[0], 1)), design]), columns=names)
    out[TARGET_COL] = y.values
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"\n已导出工程化数据集到：{path}")

def main():
    raw = fetch_dataframe()
    print(f"取数完成：{raw.shape[0]} 行，{raw.shape[1]} 列")

    df = make_derived_features(raw)
    X, y, feat_cols = build_feature_target(df)

    print(f"建模样本：X={X.shape}, y={y.shape}")
    print(f"使用的离散自变量（独热）：{feat_cols}")

    result = fit_fractional_logit(X, y)

    print("\n=== GLM Binomial（Fractional Logit）摘要 ===")
    print(result["glm"].summary(xname=result["feature_names"]))

    print("\n=== 评估指标（测试集） ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v:.4f}")

    if EXPORT_ENGINEERED_CSV:
        export_engineered(result["preproc"], X, y, ENGINEERED_CSV_PATH)

if __name__ == "__main__":
    main()
