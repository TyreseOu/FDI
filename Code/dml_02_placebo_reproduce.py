#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from econml.dml import LinearDML
from econml.inference import StatsModelsInference

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

PATH = "Dataset.csv"
LOG_DIR = "dml_placebo"

ID_COL = "市级行政代码"
YEAR_COL = "年份"

N_SPLITS = 5
SEED = 21
USE_LN1P_Y = True
WINSOR_P = 0.01
LAG_K = 1
B = 10
PERMUTE_WITHIN_CITY = True

MODEL_POOL = ["RF", "GBDT", "XGB", "LGB", "Ridge", "CAT"]

Y_COLS = [
    "外商直接投资（百万美元）",
    "外商直接投资额（百万元人民币）",
]

X_CONTROLS = [
    "中等职业教育学校在校学生数（人）全市",
    "地区生产总值增长率-全市(%)",
    "地方一般公共预算收入（万元）全市",
    "城镇非私营单位在岗职工平均工资（元）",
    "普通小学专任教师数（人）全市",
    "工业二氧化硫排放量（吨）全市",
    "普通中学（所）全市",
    "普通高等学校专任教师数（人）全市",
    "污水处理厂集中处理率（%）全市",
    "电信业务收入（万元）全市",
    "第一产业增加值(万元)",
    "第二产业增加值(万元)",
    "城镇职工基本医疗保险参保人数（人）全市",
    "规模以上工业企业主要财务指标_利润总额（万元）全市",
    "规模以上工业企业数（个）全市",
    "邮政业务收入（万元）全市",
    "限额以上批发零售业商品销售总额（万元）全市",
    "限额以上批发零售商贸企业数_法人数（个）全市",
]

TREATMENTS = [
    "独任审理率",
    "人均律师数量",
    "一审陪审率",
    "二审开庭率",
    "案件平均审理天数",
    "案件数占人口比例",
    "上诉率",
    "再审率",
    "裁判文书说理平均字数",
    "一审上诉改判率",
    "一审上诉发回重审率",
    "生效案件改判率",
    "案件发回重审率",
]


def log_step(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_models_arg(s):
    if s is None:
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def reorder_models_last_xgb_lgb(models):
    def _key(m):
        u = str(m).upper()
        if u == "XGB":
            return (1, 0)
        if u == "LGB":
            return (1, 1)
        return (0, 0)

    return sorted(models, key=_key)


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "外商直接投资额（百万人民币）" in df.columns and "外商直接投资额（百万元人民币）" not in df.columns:
        rename_map["外商直接投资额（百万人民币）"] = "外商直接投资额（百万元人民币）"
    if "外商直接投资额(百万人民币)" in df.columns and "外商直接投资额（百万元人民币）" not in df.columns:
        rename_map["外商直接投资额(百万人民币)"] = "外商直接投资额（百万元人民币）"
    if "标的额占人口比例" in df.columns and "审理费占人口比例" not in df.columns:
        rename_map["标的额占人口比例"] = "审理费占人口比例"

    alias_controls = {
        "中等职业教育学校在校学生数(人)": "中等职业教育学校在校学生数（人）全市",
        "地区生产总值增长率(%)": "地区生产总值增长率-全市(%)",
        "地方财政一般预算内收入(万元)": "地方一般公共预算收入（万元）全市",
        "小学专任教师数(人)": "普通小学专任教师数（人）全市",
        "工业二氧化硫排放量(吨)": "工业二氧化硫排放量（吨）全市",
        "普通中学学校数(所)": "普通中学（所）全市",
        "普通高等学校专任教师数(人)": "普通高等学校专任教师数（人）全市",
        "污水处理厂集中处理率(%)": "污水处理厂集中处理率（%）全市",
        "电信业务收入(万元)": "电信业务收入（万元）全市",
        "职工基本医疗保险参保人数(人)": "城镇职工基本医疗保险参保人数（人）全市",
        "规模以上工业企业利润总额(万元)": "规模以上工业企业主要财务指标_利润总额（万元）全市",
        "规模以上工业企业数(个)": "规模以上工业企业数（个）全市",
        "邮政业务收入(万元)": "邮政业务收入（万元）全市",
        "限额以上批发零售业商品销售总额(万元)": "限额以上批发零售业商品销售总额（万元）全市",
        "限额以上批发零售业法人企业数(个)": "限额以上批发零售商贸企业数_法人数（个）全市",
    }
    for old, new in alias_controls.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new

    return df.rename(columns=rename_map) if rename_map else df


def winsorize(s: pd.Series, p: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def add_lag(df: pd.DataFrame, cols, k: int) -> pd.DataFrame:
    if k <= 0:
        return df
    for c in cols:
        df[c] = df.groupby(ID_COL)[c].shift(k)
    return df


def get_float_attr_or_method(obj, name):
    if not hasattr(obj, name):
        return np.nan
    v = getattr(obj, name)
    if callable(v):
        v = v()
    try:
        return float(v)
    except Exception:
        try:
            return float(np.asarray(v).reshape(-1)[0])
        except Exception:
            return np.nan


def permute_within_group(values_1d, groups_1d, rng):
    out = values_1d.copy()
    for g in pd.unique(groups_1d):
        idx = np.where(groups_1d == g)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def build_nuisance_model(model_name: str, seed: int):
    name = str(model_name).upper()
    if name == "RF":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    if name == "GBDT":
        return GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, random_state=seed)
    if name == "RIDGE":
        return Ridge(alpha=1.0)
    if name == "XGB":
        return XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="reg:squarederror", random_state=seed, n_jobs=-1
        )
    if name == "LGB":
        return LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0,
            random_state=seed, n_jobs=-1
        )
    if name == "CAT":
        return CatBoostRegressor(
            iterations=1200, learning_rate=0.03, depth=6,
            loss_function="RMSE", random_seed=seed, verbose=False
        )
    raise ValueError(f"不支持的模型: {model_name}")


def get_available_model_pool(requested_models=None):
    pool = []
    for m in MODEL_POOL:
        mu = str(m).upper()
        if mu == "XGB" and XGBRegressor is None:
            log_step("[WARN] xgboost 未安装，跳过 XGB")
            continue
        if mu == "LGB" and LGBMRegressor is None:
            log_step("[WARN] lightgbm 未安装，跳过 LGB")
            continue
        if mu == "CAT" and CatBoostRegressor is None:
            log_step("[WARN] catboost 未安装，跳过 CAT")
            continue
        pool.append(m)

    if requested_models:
        req = {x.upper() for x in requested_models}
        pool = [m for m in pool if str(m).upper() in req]

    if not pool:
        raise ValueError("可用模型池为空，请检查 --models 或依赖安装。")
    return pool


def fit_dml_and_get_ate(Y, T, X, groups, cv, seed, model_name):
    est = LinearDML(
        model_y=build_nuisance_model(model_name, seed),
        model_t=build_nuisance_model(model_name, seed),
        cv=cv,
        random_state=seed,
    )
    try:
        est.fit(Y, T.reshape(-1, 1), X=X, groups=groups, inference=StatsModelsInference())
    except TypeError:
        est.fit(Y, T.reshape(-1, 1), X=X, groups=groups, inference="statsmodels")

    ate_inf = est.ate_inference(X=X)
    ate = get_float_attr_or_method(ate_inf, "mean_point")
    se = get_float_attr_or_method(ate_inf, "stderr_mean")
    z = get_float_attr_or_method(ate_inf, "zstat")
    p = get_float_attr_or_method(ate_inf, "pvalue")
    return ate, se, z, p


def upsert_by_model(csv_path: str, new_df: pd.DataFrame):
    if new_df is None or new_df.empty:
        return
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            old_df = pd.DataFrame()
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        out_df = new_df.copy()
    else:
        for c in new_df.columns:
            if c not in old_df.columns:
                old_df[c] = np.nan
        for c in old_df.columns:
            if c not in new_df.columns:
                new_df[c] = np.nan
        old_df = old_df[new_df.columns]
        done_models = set(new_df["model"].astype(str).unique())
        old_df = old_df.loc[~old_df["model"].astype(str).isin(done_models)].copy()
        out_df = pd.concat([old_df, new_df], ignore_index=True)

    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main():
    global PATH, LOG_DIR, B, N_SPLITS, SEED, PERMUTE_WITHIN_CITY

    parser = argparse.ArgumentParser(description="DML + Placebo（多模型、可续跑、实时日志）")
    parser.add_argument("--path", default=PATH)
    parser.add_argument("--out-dir", default=LOG_DIR)
    parser.add_argument("--models", default=None, help="逗号分隔，如 RF,GBDT,XGB,LGB,Ridge,CAT")
    parser.add_argument("--B", type=int, default=B)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--permute-within-city", type=int, default=1)
    parser.add_argument("--resume-existing", type=int, default=1, help="1=续跑并跳过已完成模型")
    args = parser.parse_args()

    PATH = args.path
    LOG_DIR = args.out_dir
    B = int(args.B)
    N_SPLITS = int(args.n_splits)
    SEED = int(args.seed)
    PERMUTE_WITHIN_CITY = bool(int(args.permute_within_city))

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        out_dir = LOG_DIR
    except PermissionError:
        out_dir = os.path.abspath("./dml_log_02")
        os.makedirs(out_dir, exist_ok=True)
        log_step(f"[WARN] 无权限写入目录: {LOG_DIR}")
        log_step(f"[WARN] 已自动回退到: {out_dir}")

    log_step(f"[INFO] 读取数据: {PATH}")
    log_step(f"[INFO] 输出目录: {out_dir}")
    log_step(
        f"[INFO] 参数: models={args.models}, B={B}, n_splits={N_SPLITS}, seed={SEED}, permute_within_city={PERMUTE_WITHIN_CITY}, resume={args.resume_existing}")

    df = pd.read_csv(PATH, encoding="utf-8-sig")
    df = align_columns(df)
    df = df.sort_values([ID_COL, YEAR_COL])

    missing_required = [c for c in [ID_COL, YEAR_COL] + Y_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"必要列缺失: {missing_required}")

    miss_controls = [c for c in X_CONTROLS if c not in df.columns]
    miss_treats = [c for c in TREATMENTS if c not in df.columns]
    if miss_controls:
        log_step(f"[WARN] 下列控制变量缺失，已跳过: {miss_controls}")
    if miss_treats:
        log_step(f"[WARN] 下列 treatment 缺失，已跳过: {miss_treats}")

    controls = [c for c in X_CONTROLS if c in df.columns]
    treats = [c for c in TREATMENTS if c in df.columns]
    if not controls:
        raise ValueError("可用控制变量为空，无法运行。")
    if not treats:
        raise ValueError("可用treatment为空，无法运行。")

    all_num = controls + treats + Y_COLS
    for c in all_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = winsorize(df[c], WINSOR_P)

    if USE_LN1P_Y:
        for y in Y_COLS:
            df[y] = np.log1p(df[y].clip(lower=0))

    df = add_lag(df, controls + treats, LAG_K)
    df = df.dropna(subset=controls + treats + Y_COLS).copy()

    groups = df[ID_COL].astype(str).values
    cv = GroupKFold(n_splits=N_SPLITS)
    rng = np.random.RandomState(SEED)

    requested_models = parse_models_arg(args.models)
    available_models = get_available_model_pool(requested_models=requested_models)
    available_models = reorder_models_last_xgb_lgb(available_models)
    log_step(f"[INFO] 模型池(可用): {available_models}")

    out_summary = os.path.join(out_dir, f"DML_placebo_summary_city_generalization_lag{LAG_K}.csv")
    out_perm = os.path.join(out_dir, f"DML_placebo_permATE_city_generalization_lag{LAG_K}.csv")

    done_models = set()
    if int(args.resume_existing) == 1 and os.path.exists(out_summary):
        try:
            old = pd.read_csv(out_summary, encoding="utf-8-sig")
            if "model" in old.columns:
                done_models = set(old["model"].astype(str).unique())
        except Exception:
            pass
    if done_models:
        log_step(f"[INFO] 续跑模式，已完成模型: {sorted(done_models)}")

    for mi, model_name in enumerate(available_models, start=1):
        if int(args.resume_existing) == 1 and str(model_name) in done_models:
            log_step(f"[SKIP] 模型已完成，跳过: {model_name}")
            continue

        log_step(f"[RUN] 模型进度 {mi}/{len(available_models)}: {model_name}")
        summary_rows = []
        perm_rows = []

        for y in Y_COLS:
            Y = df[y].values
            X = df[controls].values
            log_step(f"[RUN] model={model_name}, y={y}, treatment数={len(treats)}, 样本数={len(Y)}")

            for ti, t in enumerate(treats, start=1):
                log_step(f"[STEP] model={model_name}, y={y}, T进度={ti}/{len(treats)}: {t}")
                T_real = df[t].values

                try:
                    ate_real, se_real, z_real, p_real = fit_dml_and_get_ate(
                        Y=Y, T=T_real, X=X, groups=groups, cv=cv, seed=SEED, model_name=model_name
                    )
                except Exception as e:
                    log_step(f"[SKIP] 真实DML失败: model={model_name}, y={y}, T={t}, err={repr(e)}")
                    continue

                ate_perm_list = []
                for b in range(B):
                    T_perm = permute_within_group(T_real, groups, rng) if PERMUTE_WITHIN_CITY else rng.permutation(
                        T_real)
                    try:
                        ate_b, _, _, _ = fit_dml_and_get_ate(
                            Y=Y, T=T_perm, X=X, groups=groups, cv=cv, seed=SEED, model_name=model_name
                        )
                    except Exception:
                        ate_b = np.nan
                    ate_perm_list.append(ate_b)
                    perm_rows.append({"model": model_name, "y": y, "treatment": t, "b": b, "ATE_perm": ate_b})

                    if b == 0 or (b + 1) % 5 == 0 or (b + 1) == B:
                        log_step(f"[PLACEBO] model={model_name}, y={y}, T={t}, 置换进度 {b + 1}/{B}")

                ate_perm = np.array(ate_perm_list, dtype=float)
                valid = np.isfinite(ate_perm)
                placebo_p = float(np.mean(np.abs(ate_perm[valid]) >= np.abs(ate_real))) if valid.any() else np.nan

                summary_rows.append(
                    {
                        "model": model_name,
                        "y": y,
                        "treatment": t,
                        "ATE_real": ate_real,
                        "SE_real": se_real,
                        "z_real": z_real,
                        "p_real_econml": p_real,
                        "B": B,
                        "placebo_p": placebo_p,
                        "perm_mean": float(np.nanmean(ate_perm)),
                        "perm_std": float(np.nanstd(ate_perm)),
                        "perm_q025": float(np.nanquantile(ate_perm, 0.025)) if valid.any() else np.nan,
                        "perm_q975": float(np.nanquantile(ate_perm, 0.975)) if valid.any() else np.nan,
                        "N": int(len(Y)),
                        "permute_within_city": bool(PERMUTE_WITHIN_CITY),
                        "lag_k": int(LAG_K),
                    }
                )

                log_step(
                    f"[DML+Placebo] model={model_name} | y={y} | T={t} | ATE={ate_real:.4f} | "
                    f"econml_p={p_real:.4g} | placebo_p={placebo_p:.4g}"
                )

        summary_df_model = pd.DataFrame(summary_rows)
        perm_df_model = pd.DataFrame(perm_rows)

        upsert_by_model(out_summary, summary_df_model)
        upsert_by_model(out_perm, perm_df_model)
        log_step(f"[OK] 模型{model_name}完成并落盘: {out_summary}")
        log_step(f"[OK] 模型{model_name}完成并落盘: {out_perm}")

    log_step(f"[DONE] 全部完成，结果在: {out_dir}")


if __name__ == "__main__":
    main()
