import os
import sys
import warnings
warnings.filterwarnings("ignore")
import configparser

import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import errorcode

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import statsmodels.api as sm

# ========= 配置区（按需修改） =========
DB_CFG = {
    "host":     os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PWD",  "password"),
    "database": os.getenv("MYSQL_DB",   "config_db"),
    "charset":  "utf8mb4",
}
inipath = Path(__file__).parent.parent / "database" / "config.ini"


TABLE_EVAL = "evalution"   # 评估表名固定
TARGET_COL = "f1"             # 因变量是比例/概率
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 可选：是否把 theta/phi 做分箱（大量离散档位时有用，默认关闭）
BIN_THETA_PHI = False
THETA_BIN_DEG = 5
PHI_BIN_DEG = 5

EXPORT_ENGINEERED_CSV = False
ENGINEERED_CSV_PATH = "engineered_dataset_onehot.csv"
# ====================================

def load_db_config(inipath):
    inipath = str(inipath)
    if not os.path.isfile(inipath):
        print(f"配置文件不存在：{inipath}，使用环境变量或默认值")
        return
    cfg = configparser.ConfigParser()
    cfg.read(inipath,encoding="utf-8")
    sections = []
    for sec in cfg.sections():
        dict_sec = {key.lower():val.split() for key,val in cfg[sec].items()}
        for key in ["port","user","password","database","host"]:
            if key not in dict_sec:
                print(f"配置文件缺少 {sec} 节的 {key} 项，使用环境变量或默认值")
                continue
        dict_sec["port"] = int(dict_sec["port"])
        dict_sec["__name__"] = sec
        sections.append(dict_sec)
    if not sections:
        print(f"配置文件无有效节，使用环境变量或默认值")
        return
    return sections
def get_conn(sections):
    name = sections.get("__name__","default")
    host = sections.get("host")
    port = sections.get("port",3306)
    user = sections.get("user")
    password = sections.get("password")
    database = sections.get("database")
    try:
        conn = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=False,
    )
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("数据库账户或密码错误"); sys.exit(1)
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("数据库不存在"); sys.exit(1)
        else:
            print(err); sys.exit(1)

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

    # 1) 删除角度列（与像素严格线性相关）
    df = df.drop(columns=[c for c in ["hFOVDeg", "vFOVDeg"] if c in df.columns], errors="ignore")

    # 2) 分辨率独热：先生成类别标签 "WxH"
    if {"hFOVPixels","vFOVPixels"}.issubset(df.columns):
        # 用 Int64 以兼容缺失值，再转字符串
        df["resolution_cls"] = (
            df["hFOVPixels"].astype("Int64").astype(str) + "x" +
            df["vFOVPixels"].astype("Int64").astype(str)
        )
    else:
        df["resolution_cls"] = np.nan

    # 3) 温差（离散来源，依旧离散使用）
    if {"maxTemperature","minTemperature"}.issubset(df.columns):
        df["temperature_range"] = df["maxTemperature"] - df["minTemperature"]
    else:
        df["temperature_range"] = np.nan

    # 4) 日期分解
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = dt.dt.month
        df["day_of_week"] = dt.dt.dayofweek
    else:
        df["month"] = np.nan
        df["day_of_week"] = np.nan

    # 5) time 统一为字符串
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    # 6) 可选：对 theta/phi 分箱（离散档位很多时降低列数；默认不启用）
    if BIN_THETA_PHI:
        if "theta" in df.columns:
            df["theta"] = (df["theta"] / THETA_BIN_DEG).round().astype("Int64") * THETA_BIN_DEG
        if "phi" in df.columns:
            df["phi"] = (df["phi"] / PHI_BIN_DEG).round().astype("Int64") * PHI_BIN_DEG

    # 7) 规范布尔列（ismanbo 等）为 0/1（仍会当作离散类别去独热）
    for bcol in ["ismanbo","ispanmao","ishajimi","ispanbaobao"]:
        if bcol in df.columns:
            if df[bcol].dtype == object:
                df[bcol] = df[bcol].str.lower().map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0})

    return df

def cast_all_to_categorical_strings(df: pd.DataFrame, cols) -> pd.DataFrame:
    """将给定列统一转为字符串类别（保留 NaN，便于众数填充）。"""
    df = df.copy()
    for c in cols:
        if c not in df.columns: 
            continue
        s = df[c]
        # 浮点：四舍五入避免 0.30000000004 这类问题
        if pd.api.types.is_float_dtype(s):
            df[c] = s.round(6).astype(str)
        elif pd.api.types.is_integer_dtype(s):
            df[c] = s.astype("Int64").astype(str)
        else:
            df[c] = s.astype(str)
    return df

def build_feature_target(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到因变量列：{TARGET_COL}")

    # 删除主键/泄露列
    df = df.drop(columns=[c for c in ["ConfigID","image_id"] if c in df.columns], errors="ignore")

    # 因变量：保留 [0,1]，剔除越界与缺失
    df = df.dropna(subset=[TARGET_COL])
    df = df[(df[TARGET_COL] >= 0.0) & (df[TARGET_COL] <= 1.0)]
    y = df[TARGET_COL].astype(float)

    # ===== 全离散独热：把所有自变量都当作“类别字符串”处理 =====
    candidate_features = [
        # param_sets 常见字段（按你的DDL与描述补充）
        "name", "maxTemperature", "minTemperature",
        "hFOVPixels", "vFOVPixels",  # 注意：仅用于构造 resolution_cls，不直接入模
        "hFOVDeg", "vFOVDeg",        # 会在派生阶段被删除
        "hFOVPixels", "vFOVPixels",
        "hFOVDeg", "vFOVDeg",
        "hFOVPixels", "vFOVPixels",  # 重复不影响
        "percentBlur", "percentNoise",
        "hazeModel", "rainRate", "visibility",
        "time", "date",
        "ismanbo","ispanmao","ishajimi","ispanbaobao",
        # images 补充列
        "R", "theta", "phi",
        # 派生列
        "resolution_cls", "temperature_range", "month", "day_of_week"
    ]
    # 实际存在的列（去重）
    feat_cols = [c for c in dict.fromkeys(candidate_features) if c in df.columns]

    # 我们不再直接使用 hFOVPixels/vFOVPixels （避免与 resolution_cls 共线）
    feat_cols = [c for c in feat_cols if c not in ("hFOVPixels", "vFOVPixels", "hFOVDeg", "vFOVDeg")]

    # 统一转字符串类别（包括数值/布尔）
    X = cast_all_to_categorical_strings(df, feat_cols)[feat_cols]

    return X, y, feat_cols

def fit_fractional_logit(X, y):
    """
    1) 全类别独热（drop='first' 避免虚拟变量陷阱）
    2) GLM Binomial 对比例/概率型 y 建模（含 0/1）
    """
    # 训练/测试划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 预处理：众数填充 + OneHot（全部特征）
    preproc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",                 # 避免与常数项完全共线
            sparse_output=False
        ))
    ])

    # 生成设计矩阵
    X_train_design = preproc.fit_transform(X_train)
    X_test_design  = preproc.transform(X_test)

    # statsmodels GLM Binomial（Fractional Logit）
    X_train_const = sm.add_constant(X_train_design, has_constant="add")
    glm_full = sm.GLM(y_train.values.astype(float), X_train_const, family=sm.families.Binomial()).fit()

    # 评估
    X_test_const = sm.add_constant(X_test_design, has_constant="add")
    y_prob = glm_full.predict(X_test_const)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    # Brier 与 LogLoss
    eps = 1e-15
    brier = float(np.mean((y_test.values - y_prob) ** 2))
    logloss = float(-np.mean(y_test.values * np.log(y_prob + eps) + (1 - y_test.values) * np.log(1 - y_prob + eps)))

    # McFadden 伪 R²（训练集上）
    X0 = np.ones((len(y_train), 1))
    null_fit = sm.GLM(y_train.values.astype(float), X0, family=sm.families.Binomial()).fit()
    pseudo_r2 = 1.0 - (glm_full.llf / null_fit.llf)

    # 特征名
    oh = preproc.named_steps["onehot"]
    feature_names = oh.get_feature_names_out(X.columns).tolist()
    feature_names = ["const"] + feature_names

    return {
        "preproc": preproc,
        "glm": glm_full,
        "feature_names": feature_names,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "y_prob": y_prob,
        "metrics": {"Brier": brier, "LogLoss": logloss, "McFaddenR2_train": float(pseudo_r2)}
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
    sections = load_db_config(inipath)
    conn = get_conn(sections)
    df_raw = fetch_dataframe()
    print(f"取数完成：{df_raw.shape[0]} 行，{df_raw.shape[1]} 列")

    df = make_derived_features(df_raw)
    X, y, feat_cols = build_feature_target(df)

    print(f"建模样本：X={X.shape}, y={y.shape}")
    print(f"使用的离散自变量（全部做独热）：{feat_cols}")

    result = fit_fractional_logit(X, y)

    # 输出统计摘要与指标
    print("\n=== GLM Binomial（Fractional Logit）摘要 ===")
    print(result["glm"].summary(xname=result["feature_names"]))

    print("\n=== 评估指标（测试集） ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v:.4f}")

    if EXPORT_ENGINEERED_CSV:
        # 用全量 X（而非切分）导出工程化矩阵
        export_engineered(result["preproc"], X, y, ENGINEERED_CSV_PATH)

if __name__ == "__main__":
    main()
