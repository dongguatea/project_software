import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import errorcode

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm

# ========= 配置区：请按需修改 =========
DB_Config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'asd515359',
    'database': 'config_db',
    'port' : 3306
}

TABLE_EVAL = "eval_metrics"  # 评估表固定名
TARGET_COL = "f1"
EXPORT_ENGINEERED_CSV = False  # 若要导出工程化设计矩阵为CSV，改为 True
ENGINEERED_CSV_PATH = "engineered_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ====================================

def get_conn():
    try:
        return mysql.connector.connect(**DB_Config)
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

    # 1) 分辨率与温差
    if {"hFOVPixels","vFOVPixels"}.issubset(df.columns):
        df["resolution"] = df["hFOVPixels"] * df["vFOVPixels"]
    else:
        df["resolution"] = np.nan

    if {"maxTemperature","minTemperature"}.issubset(df.columns):
        df["temperature_range"] = df["maxTemperature"] - df["minTemperature"]
    else:
        df["temperature_range"] = np.nan

    # 2) 日期分解（month, day_of_week）
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = dt.dt.month
        df["day_of_week"] = dt.dt.dayofweek
    else:
        df["month"] = np.nan
        df["day_of_week"] = np.nan

    # 3) 保证 time 是字符串（如 '0:00','12:00'）
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    # 4) 二元目标列规范（如存在）
    for bcol in ["ismanbo","ispanmao","ishajimi","ispanbaobao"]:
        if bcol in df.columns:
            # 若是字符串，映射为0/1；若已是数值则保持
            if df[bcol].dtype == object:
                df[bcol] = df[bcol].str.lower().map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0})
    return df

def build_feature_target(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"未找到因变量列：{TARGET_COL}")
    # 删除明显泄露或无用主键列
    drop_cols = [c for c in ["ConfigID","image_id","hFOVDeg","vFOVDeg"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 丢弃缺少目标值的样本
    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL].astype(float)

    # ====== 列分组（按你的语义）======
    # 数值（含二元0/1）：统一按数值流水线（填充+标准化）
    num_features = [
        # 连续原始数值
        c for c in [
            "maxTemperature","minTemperature","hFOVPixels","vFOVPixels",
            "percentBlur","percentNoise",
            "rainRate","visibility","R","theta","phi",
            # 派生
            "resolution","temperature_range"
        ] if c in df.columns
    ]

    # 二元特征（0/1）也放到数值处理里（统一尺度）
    binary_features = [c for c in ["ismanbo","ispanmao","ishajimi","ispanbaobao"] if c in df.columns]
    num_features += binary_features

    # 类别变量：独热编码
    cat_features = [c for c in ["name","hazeModel","time","month","day_of_week"] if c in df.columns]

    # 构建 X
    used_cols = list(dict.fromkeys(num_features + cat_features + [TARGET_COL]))  # 去重保持顺序
    X = df[used_cols].drop(columns=[TARGET_COL])

    return X, y, num_features, cat_features, binary_features

def make_pipeline(num_features, cat_features):
    num_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_proc, num_features),
            ("cat", cat_proc, cat_features),
        ],
        remainder="drop"
    )
    model = Pipeline(steps=[
        ("preprocess", preproc),
        ("linreg", LinearRegression())
    ])
    return model

def feature_names_after_fit(pipeline, num_features, cat_features):
    pre = pipeline.named_steps["preprocess"]
    names = []
    # 数值列（标准化不改列数）
    names.extend(num_features)
    # 类别列（独热）
    if cat_features:
        oh = pre.named_transformers_["cat"].named_steps["onehot"]
        names.extend(oh.get_feature_names_out(cat_features).tolist())
    return names

def main():
    # 1) 取数
    raw = fetch_dataframe()
    print(f"取数完成：{raw.shape[0]} 行，{raw.shape[1]} 列")

    # 2) 构造派生特征
    df = make_derived_features(raw)

    # 3) 构建特征/目标 + 列分组
    X, y, num_features, cat_features, binary_features = build_feature_target(df)
    print(f"建模样本：X={X.shape}, y={y.shape}")
    print(f"数值列（含二元）：{num_features}")
    print(f"类别列（独热）：{cat_features}")

    # 4) 切分与建模
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipe = make_pipeline(num_features, cat_features)
    pipe.fit(X_train, y_train)

    # 5) 评估
    y_pred = pipe.predict(X_test)
    print("\n=== Scikit-learn 线性回归评估 ===")
    print(f"Test R^2 : {r2_score(y_test, y_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

    # 6) OLS 摘要（与管道同一套特征工程）
    pre = pipe.named_steps["preprocess"]
    X_train_design = pre.transform(X_train)
    feat_names = feature_names_after_fit(pipe, num_features, cat_features)
    X_sm = sm.add_constant(X_train_design, has_constant="add")
    ols = sm.OLS(y_train.values.astype(float), X_sm).fit()
    print("\n=== statsmodels OLS 摘要 ===")
    print(ols.summary(xname=["const"] + feat_names))

    # 7) （可选）导出工程化后的设计矩阵
    if EXPORT_ENGINEERED_CSV:
        X_all_design = pre.transform(X)
        all_names = feature_names_after_fit(pipe, num_features, cat_features)
        out = pd.DataFrame(X_all_design, columns=all_names)
        out[TARGET_COL] = y.values
        out.to_csv(ENGINEERED_CSV_PATH, index=False, encoding="utf-8")
        print(f"\n已导出工程化数据集到：{ENGINEERED_CSV_PATH}")

if __name__ == "__main__":
    main()
