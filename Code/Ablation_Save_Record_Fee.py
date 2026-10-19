#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import warnings
from datetime import datetime
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams, font_manager

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
except Exception as e:
    raise ImportError(
        "闇€瑕?statsmodels 鎵嶈兘鍋氬閲嶆瘮杈冩牎姝ｏ紝璇峰厛瀹夎锛歱ip install statsmodels\n"
        f"import error: {repr(e)}"
    )

try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None
    _xgb_err = e

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None
    _lgb_err = e

try:
    from catboost import CatBoostRegressor
except Exception as e:
    CatBoostRegressor = None
    _cat_err = e

DEFAULT_PATH = r"Dataset.csv"
DEFAULT_LOG_DIR = r"ablation_log_01"

ID_COL = "市级行政代码"
YEAR_COL = "年份"

N_SPLITS = 5
N_REPEATS = 20
SEED = 21

ROLL_W = 2
ROLL_MIN_TRAIN_N = 50

USE_LN1P_Y = True
WINSOR_P = 0.01
LAG_K = 1
PROGRESS_LOG_EVERY = 5


def log_step(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _fmt_duration(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    seconds = int(round(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


Y_COLS = [
    "外商直接投资（百万美元）",
    "外商直接投资额（百万元人民币）",
]

X_CORE = [
    "案件总数",
    "案件数占人口比例",
    "审理费占人口比例",
    "上诉率",
    "再审率",
    "裁判文书说理平均字数",
    "一审上诉改判率",
    "一审上诉发回重审率",
    "生效案件改判率",
    "案件发回重审率",
    "一审陪审率",
    "二审开庭率",
    "独任审理率",
    "人均律师数量",
    "案件平均审理天数",
]

X_CORE_D = [
    "案件总数",
    "案件数占人口比例",
    "审理费占人口比例",
    "上诉率",
    "再审率",
    "裁判文书说理平均字数",
    "一审上诉改判率",
    "一审上诉发回重审率",
    "生效案件改判率",
    "案件发回重审率",
]

X_CORE_P = [
    "一审陪审率",
    "二审开庭率",
    "独任审理率",
    "人均律师数量",
    "案件平均审理天数",
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
    "第一产业占地区生产总值的比重-全市",
    "第二产业占地区生产总值的比重-全市",
    "城镇职工基本医疗保险参保人数（人）全市",
    "规模以上工业企业主要财务指标_利润总额（万元）全市",
    "规模以上工业企业数（个）全市",
    "邮政业务收入（万元）全市",
    "限额以上批发零售业商品销售总额（万元）全市",
    "限额以上批发零售商贸企业数_法人数（个）全市",
]

FEAT_ORDER = [
    "X_CONTROLS",
    "X_CONTROLS+X_CORE_D",
    "X_CONTROLS+X_CORE_P",
    "X_CONTROLS+X_CORE",
]
FEATURE_SETS_RAW = {
    "X_CONTROLS": X_CONTROLS,
    "X_CONTROLS+X_CORE_D": X_CONTROLS + X_CORE_D,
    "X_CONTROLS+X_CORE_P": X_CONTROLS + X_CORE_P,
    "X_CONTROLS+X_CORE": X_CONTROLS + X_CORE,
}


def normalize_known_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    if "外商直接投资额（百万人民币）" in df.columns and "外商直接投资额（百万元人民币）" not in df.columns:
        rename_map["外商直接投资额（百万人民币）"] = "外商直接投资额（百万元人民币）"
    if "外商直接投资额(百万人民币)" in df.columns and "外商直接投资额（百万元人民币）" not in df.columns:
        rename_map["外商直接投资额(百万人民币)"] = "外商直接投资额（百万元人民币）"
    if "标的额占人口比例" in df.columns and "审理费占人口比例" not in df.columns:
        rename_map["标的额占人口比例"] = "审理费占人口比例"

    return df.rename(columns=rename_map)


def build_models():
    models = {
        "Ridge": Ridge(alpha=1.0),
    }

    if CatBoostRegressor is None:
        raise ImportError(f"catboost import failed: {repr(_cat_err)}")

    models["CAT"] = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=False,
    )
    return models


def winsorize(s: pd.Series, p: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if sst == 0 else 1 - sse / sst


def make_pipe(model_template):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("m", clone(model_template)),
        ]
    )


def repeated_group_kfold(groups: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    for rep in range(n_repeats):
        perm = uniq.copy()
        rng.shuffle(perm)
        mp = {g: i for i, g in enumerate(perm)}
        mapped = np.array([mp[g] for g in groups])
        gkf = GroupKFold(n_splits=n_splits)
        fold = 0
        for tr, te in gkf.split(np.zeros(len(groups)), groups=mapped):
            yield rep, fold, tr, te
            fold += 1


def rolling_years(unique_years_sorted: np.ndarray, w: int):
    yrs = np.array(sorted(unique_years_sorted))
    for i in range(w, len(yrs)):
        t = int(yrs[i])
        train_years = set(int(x) for x in yrs[i - w: i])
        yield t, train_years


def adjust_within_group(df_in, p_col, group_cols, method):
    out = df_in.copy()
    out[p_col + f"_{method}"] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        idx = list(idx)
        pvals = out.loc[idx, p_col].values.astype(float)
        mask = np.isfinite(pvals)
        if mask.sum() == 0:
            continue
        adj = np.full_like(pvals, np.nan, dtype=float)
        adj[mask] = multipletests(pvals[mask], method=method)[1]
        out.loc[idx, p_col + f"_{method}"] = adj
    return out


def run_experiment(exp_name: str, df: pd.DataFrame, models: dict, feature_sets: dict):
    fold_rows = []
    summary_rows = []

    log_step(f"[{exp_name}] 实验开始：模型数={len(models)}，特征集数={len(feature_sets)}")

    for model_name, model_template in models.items():
        log_step(f"[{exp_name}] 开始模型: {model_name}")
        for y in Y_COLS:
            dfy = df.dropna(subset=[y]).copy()
            if exp_name == "roll":
                dfy = dfy.dropna(subset=[YEAR_COL]).copy()
            if dfy.empty:
                log_step(f"[{exp_name}] 模型={model_name} 目标={y} 无可用样本，跳过")
                continue

            yv = dfy[y].values
            x_dict = {fs_name: dfy[cols] for fs_name, cols in feature_sets.items()}

            if exp_name == "city":
                g = dfy[ID_COL].astype(str).values
                split_plan = list(repeated_group_kfold(g, N_SPLITS, N_REPEATS, SEED))
            elif exp_name == "roll":
                uniq_years = dfy[YEAR_COL].dropna().unique()
                split_plan = list(rolling_years(uniq_years, ROLL_W))
            else:
                raise ValueError("exp_name must be 'city' or 'roll'")

            total_splits = len(split_plan)
            task_start = time.perf_counter()
            log_step(
                f"[{exp_name}] 模型={model_name} 目标={y} 样本数={len(dfy)} split数={total_splits} 特征集={list(feature_sets.keys())}"
            )

            r2_map = {fs: [] for fs in feature_sets}
            mae_map = {fs: [] for fs in feature_sets}

            if exp_name == "city":
                for split_idx, (rep, fold, tr, te) in enumerate(split_plan, start=1):
                    split_id = int(rep * 1000 + fold)
                    for fs_name in feature_sets:
                        pipe = make_pipe(model_template)
                        pipe.fit(x_dict[fs_name].iloc[tr], yv[tr])
                        pred = pipe.predict(x_dict[fs_name].iloc[te])
                        r2 = oos_r2(yv[te], pred)
                        mae = mean_absolute_error(yv[te], pred)
                        r2_map[fs_name].append(r2)
                        mae_map[fs_name].append(mae)
                        fold_rows.append(
                            {
                                "exp": "city",
                                "model": model_name,
                                "y": y,
                                "feature_set": fs_name,
                                "split_id": split_id,
                                "rep": int(rep),
                                "fold": int(fold),
                                "test_year": np.nan,
                                "R2": float(r2) if np.isfinite(r2) else np.nan,
                                "MAE": float(mae) if np.isfinite(mae) else np.nan,
                                "n_test": int(len(te)),
                            }
                        )
                    if split_idx == 1 or split_idx % PROGRESS_LOG_EVERY == 0 or split_idx == total_splits:
                        elapsed = time.perf_counter() - task_start
                        eta = (elapsed / split_idx) * (total_splits - split_idx) if split_idx > 0 else np.nan
                        log_step(
                            f"[{exp_name}] 模型={model_name} 目标={y} split进度 {split_idx}/{total_splits} "
                            f"(rep={rep}, fold={fold}) | 已耗时={_fmt_duration(elapsed)} | ETA={_fmt_duration(eta)}"
                        )
            else:
                years_series = dfy[YEAR_COL].values.astype(int)
                for split_idx, (t, train_years) in enumerate(split_plan, start=1):
                    te = np.where(years_series == int(t))[0]
                    tr = np.where(np.isin(years_series, list(train_years)))[0]
                    if len(te) == 0 or len(tr) == 0:
                        if split_idx == 1 or split_idx % PROGRESS_LOG_EVERY == 0 or split_idx == total_splits:
                            elapsed = time.perf_counter() - task_start
                            eta = (elapsed / split_idx) * (total_splits - split_idx) if split_idx > 0 else np.nan
                            log_step(
                                f"[{exp_name}] 模型={model_name} 目标={y} split进度 {split_idx}/{total_splits} (test_year={t}) 跳过：train/test为空"
                                f" | 已耗时={_fmt_duration(elapsed)} | ETA={_fmt_duration(eta)}"
                            )
                        continue
                    if (ROLL_MIN_TRAIN_N is not None) and (len(tr) < ROLL_MIN_TRAIN_N):
                        if split_idx == 1 or split_idx % PROGRESS_LOG_EVERY == 0 or split_idx == total_splits:
                            elapsed = time.perf_counter() - task_start
                            eta = (elapsed / split_idx) * (total_splits - split_idx) if split_idx > 0 else np.nan
                            log_step(
                                f"[{exp_name}] 模型={model_name} 目标={y} split进度 {split_idx}/{total_splits} (test_year={t}) 跳过：训练样本不足{len(tr)}"
                                f" | 已耗时={_fmt_duration(elapsed)} | ETA={_fmt_duration(eta)}"
                            )
                        continue
                    split_id = int(t)
                    for fs_name in feature_sets:
                        pipe = make_pipe(model_template)
                        pipe.fit(x_dict[fs_name].iloc[tr], yv[tr])
                        pred = pipe.predict(x_dict[fs_name].iloc[te])
                        r2 = oos_r2(yv[te], pred)
                        mae = mean_absolute_error(yv[te], pred)
                        r2_map[fs_name].append(r2)
                        mae_map[fs_name].append(mae)
                        fold_rows.append(
                            {
                                "exp": "roll",
                                "model": model_name,
                                "y": y,
                                "feature_set": fs_name,
                                "split_id": split_id,
                                "rep": np.nan,
                                "fold": np.nan,
                                "test_year": int(t),
                                "R2": float(r2) if np.isfinite(r2) else np.nan,
                                "MAE": float(mae) if np.isfinite(mae) else np.nan,
                                "n_test": int(len(te)),
                            }
                        )
                    if split_idx == 1 or split_idx % PROGRESS_LOG_EVERY == 0 or split_idx == total_splits:
                        elapsed = time.perf_counter() - task_start
                        eta = (elapsed / split_idx) * (total_splits - split_idx) if split_idx > 0 else np.nan
                        log_step(
                            f"[{exp_name}] 模型={model_name} 目标={y} split进度 {split_idx}/{total_splits} (test_year={t})"
                            f" | 已耗时={_fmt_duration(elapsed)} | ETA={_fmt_duration(eta)}"
                        )

            base_r2 = float(np.nanmean(np.asarray(r2_map["X_CONTROLS"], dtype=float)))
            base_mae = float(np.nanmean(np.asarray(mae_map["X_CONTROLS"], dtype=float)))

            for fs_name in feature_sets.keys():
                r2_arr = np.asarray(r2_map[fs_name], dtype=float)
                mae_arr = np.asarray(mae_map[fs_name], dtype=float)
                mean_r2 = float(np.nanmean(r2_arr))
                mean_mae = float(np.nanmean(mae_arr))
                summary_rows.append(
                    {
                        "exp": exp_name,
                        "model": model_name,
                        "y": y,
                        "feature_set": fs_name,
                        "mean_R2": mean_r2,
                        "median_R2": float(np.nanmedian(r2_arr)),
                        "std_R2": float(np.nanstd(r2_arr, ddof=0)),
                        "mean_MAE": mean_mae,
                        "median_MAE": float(np.nanmedian(mae_arr)),
                        "std_MAE": float(np.nanstd(mae_arr, ddof=0)),
                        "N_eval": int(np.isfinite(r2_arr).sum()),
                        "base_R2": base_r2,
                        "base_MAE": base_mae,
                        "dR2_vs_controls": float(mean_r2 - base_r2),
                        "dMAE_vs_controls": float(mean_mae - base_mae),
                    }
                )
            log_step(
                f"[{exp_name}] 模型={model_name} 目标={y} 完成：基线R2={base_r2:.4f} 基线MAE={base_mae:.4f}"
                f" | 总耗时={_fmt_duration(time.perf_counter() - task_start)}"
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["y", "model", "feature_set"]).reset_index(drop=True)

    comparisons = [
        ("X_CONTROLS+X_CORE", "X_CONTROLS"),
        ("X_CONTROLS+X_CORE_D", "X_CONTROLS"),
        ("X_CONTROLS+X_CORE_P", "X_CONTROLS"),
    ]

    test_rows = []
    for model_name in fold_df["model"].unique():
        for yname in fold_df["y"].unique():
            sub = fold_df[(fold_df["model"] == model_name) & (fold_df["y"] == yname)].copy()
            if sub.empty:
                continue

            pv_r2 = sub.pivot_table(index="split_id", columns="feature_set", values="R2", aggfunc="mean")
            pv_mae = sub.pivot_table(index="split_id", columns="feature_set", values="MAE", aggfunc="mean")

            for bset, aset in comparisons:
                if (aset not in pv_r2.columns) or (bset not in pv_r2.columns):
                    continue

                d_r2 = (pv_r2[bset] - pv_r2[aset]).dropna().values.astype(float)
                n = int(len(d_r2))
                p_t_r2 = stats.ttest_1samp(d_r2, popmean=0.0, alternative="greater").pvalue if n > 1 else np.nan
                p_w_r2 = stats.wilcoxon(d_r2, alternative="greater").pvalue if (n > 0 and np.any(d_r2 != 0)) else np.nan

                d_mae = (pv_mae[bset] - pv_mae[aset]).dropna().values.astype(float)
                n2 = int(len(d_mae))
                p_t_mae = stats.ttest_1samp(d_mae, popmean=0.0, alternative="less").pvalue if n2 > 1 else np.nan
                p_w_mae = stats.wilcoxon(d_mae, alternative="less").pvalue if (
                            n2 > 0 and np.any(d_mae != 0)) else np.nan

                test_rows.append(
                    {
                        "exp": exp_name,
                        "model": model_name,
                        "y": yname,
                        "compare": f"{bset} vs {aset}",
                        "n_splits": n,
                        "mean_dR2": float(np.mean(d_r2)) if n else np.nan,
                        "p_t_R2": float(p_t_r2) if np.isfinite(p_t_r2) else np.nan,
                        "p_w_R2": float(p_w_r2) if np.isfinite(p_w_r2) else np.nan,
                        "mean_dMAE": float(np.mean(d_mae)) if n2 else np.nan,
                        "p_t_MAE": float(p_t_mae) if np.isfinite(p_t_mae) else np.nan,
                        "p_w_MAE": float(p_w_mae) if np.isfinite(p_w_mae) else np.nan,
                    }
                )

    tests_df = pd.DataFrame(test_rows).sort_values(["y", "model", "compare"]).reset_index(drop=True)
    group_cols = ["exp", "model", "y"]
    for col in ["p_t_R2", "p_w_R2", "p_t_MAE", "p_w_MAE"]:
        tests_df = adjust_within_group(tests_df, col, group_cols, method="holm")
        tests_df = adjust_within_group(tests_df, col, group_cols, method="fdr_bh")

    log_step(f"[{exp_name}] 实验结束：fold行数={len(fold_df)} summary行数={len(summary_df)} tests行数={len(tests_df)}")
    return fold_df, summary_df, tests_df


def safe_name(s: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = str(s)
    for b in bad:
        out = out.replace(b, "_")
    return out


def set_plot_style() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    chosen = [f for f in candidates if f in installed]
    if not chosen:
        chosen = ["DejaVu Sans"]
    rcParams["font.sans-serif"] = chosen
    rcParams["axes.unicode_minus"] = False
    rcParams["font.size"] = 11
    rcParams["axes.titlesize"] = 12
    rcParams["axes.labelsize"] = 11
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10
    rcParams["figure.dpi"] = 150


def color_shades(base_color: str, n: int):
    rgb = mcolors.to_rgb(base_color)
    if n <= 1:
        return [rgb]
    return [
        (
            rgb[0] * (0.6 + 0.4 * i / (n - 1)),
            rgb[1] * (0.6 + 0.4 * i / (n - 1)),
            rgb[2] * (0.6 + 0.4 * i / (n - 1)),
        )
        for i in range(n)
    ]


def plot_r2_barh_sci(df_sub: pd.DataFrame, model_name: str, y_name: str, fig_dir: str, xmin: float = 0.5):
    d = df_sub[df_sub["y"] == y_name].sort_values("feature_set").copy()
    if d.empty:
        return

    base_colors = {"usd": "#4C72B0", "cny": "#55A868"}
    is_usd = ("美元" in y_name)
    key = "usd" if is_usd else "cny"
    colors = color_shades(base_colors[key], len(d))
    y_short = "FDI(美元)" if is_usd else "FDI(人民币)"

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.barh(d["feature_label"], d["mean_R2"], color=colors)
    if str(model_name).strip().lower() == "ridge":
        xmin_cur = max(0.0, float(d["mean_R2"].min()) - 0.03)
        xmax_cur = max(float(d["mean_R2"].max()) + 0.02, xmin_cur + 0.08)
    else:
        xmin_cur = xmin
        xmax_cur = max(float(d["mean_R2"].max()) + 0.02, xmin_cur + 0.08)
    ax.set_xlim(xmin_cur, xmax_cur)
    ax.set_xlabel(r"Out-of-sample $R^2$")
    ax.set_title(f"{model_name}｜{y_short}")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(d["mean_R2"].values):
        ax.text(float(v) + 0.003, i, f"{float(v):.3f}", va="center", fontsize=10)

    plt.tight_layout()
    safe_model = safe_name(model_name)
    safe_y = "usd" if is_usd else "cny"
    out_png = os.path.join(fig_dir, f"{safe_model}_{safe_y}_r2.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fmt_num(x, nd=3):
    if pd.isna(x) or not np.isfinite(x):
        return ""
    return f"{float(x):.{nd}f}"


def fmt_p(x):
    if pd.isna(x) or not np.isfinite(x):
        return ""
    x = float(x)
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def export_city_plots_and_tables(summary_df: pd.DataFrame, tests_df: pd.DataFrame, log_dir: str) -> None:
    if summary_df.empty:
        return
    set_plot_style()

    fig_dir = os.path.join(log_dir, "figs_01")
    table_dir = os.path.join(log_dir, "tables_01")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    feat_order = [
        "X_CONTROLS",
        "X_CONTROLS+X_CORE_D",
        "X_CONTROLS+X_CORE_P",
        "X_CONTROLS+X_CORE",
    ]
    feat_label_map = {
        "X_CONTROLS": "控制",
        "X_CONTROLS+X_CORE_D": "+分配正义",
        "X_CONTROLS+X_CORE_P": "+程序正义",
        "X_CONTROLS+X_CORE": "+全部司法",
    }

    df = summary_df.copy()
    df["feature_set"] = pd.Categorical(df["feature_set"], categories=feat_order, ordered=True)
    df["feature_label"] = df["feature_set"].map(feat_label_map)

    models = list(df["model"].dropna().unique())
    for m in models:
        df_m = df[df["model"] == m].copy()
        if not df_m.empty:
            y_usd = next((x for x in df_m["y"].unique() if "美元" in str(x)), None)
            y_cny = next((x for x in df_m["y"].unique() if ("人民币" in str(x) or "百万元人民币" in str(x))), None)
            if y_usd:
                plot_r2_barh_sci(df_m, model_name=m, y_name=y_usd, fig_dir=fig_dir, xmin=0.5)
            if y_cny:
                plot_r2_barh_sci(df_m, model_name=m, y_name=y_cny, fig_dir=fig_dir, xmin=0.5)

    if tests_df.empty:
        return

    pcols_core = [
        "p_t_R2", "p_w_R2", "p_t_MAE", "p_w_MAE",
        "p_t_R2_holm", "p_w_R2_holm", "p_t_MAE_holm", "p_w_MAE_holm",
        "p_t_R2_fdr_bh", "p_w_R2_fdr_bh", "p_t_MAE_fdr_bh", "p_w_MAE_fdr_bh",
    ]
    pcols = [c for c in pcols_core if c in tests_df.columns]
    keep_cols = ["model", "y", "compare", "n_splits", "mean_dR2", "mean_dMAE"] + pcols
    df_show = tests_df[keep_cols].copy()

    df_show["mean_dR2"] = df_show["mean_dR2"].apply(lambda v: fmt_num(v, 3))
    df_show["mean_dMAE"] = df_show["mean_dMAE"].apply(lambda v: fmt_num(v, 3))
    for c in pcols:
        df_show[c] = df_show[c].apply(fmt_p)
    df_show = df_show.sort_values(["y", "model", "compare"]).reset_index(drop=True)

    r2_cols = ["model", "y", "compare", "n_splits", "mean_dR2"] + [c for c in pcols if "R2" in c]
    mae_cols = ["model", "y", "compare", "n_splits", "mean_dMAE"] + [c for c in pcols if "MAE" in c]
    df_r2 = df_show[r2_cols].copy()
    df_mae = df_show[mae_cols].copy()

    df_show.to_csv(os.path.join(table_dir, "city_tests_pvalues_full.csv"), index=False, encoding="utf-8-sig")
    df_r2.to_csv(os.path.join(table_dir, "city_tests_pvalues_r2.csv"), index=False, encoding="utf-8-sig")
    df_mae.to_csv(os.path.join(table_dir, "city_tests_pvalues_mae.csv"), index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="01 鏁版嵁娑堣瀺+淇濆瓨璁板綍锛堝鐞嗚垂鐗堬級")
    parser.add_argument("--path", default=DEFAULT_PATH, help="杈撳叆鏁版嵁闆咰SV")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="杈撳嚭鐩綍")
    args = parser.parse_args()

    try:
        os.makedirs(args.log_dir, exist_ok=True)
        log_dir = args.log_dir
    except PermissionError:
        fallback = os.path.abspath("./ablation_log_01")
        os.makedirs(fallback, exist_ok=True)
        log_step(f"[WARN] 无权限写入目录: {args.log_dir}")
        log_step(f"[WARN] 已自动回退到: {fallback}")
        log_dir = fallback

    log_step(f"[INFO] 读取数据: {args.path}")
    df = pd.read_csv(args.path, encoding="utf-8-sig")
    df = normalize_known_aliases(df)
    log_step(f"[INFO] 原始数据行数: {len(df)}，列数: {len(df.columns)}")

    if ID_COL not in df.columns:
        for c in df.columns:
            cs = str(c)
            if ("市级" in cs and "代码" in cs) or ("城市" in cs and "代码" in cs) or ("甯" in cs and "浠ｇ爜" in cs):
                df = df.rename(columns={c: ID_COL})
                break
    if YEAR_COL not in df.columns:
        for c in df.columns:
            cs = str(c).strip().lower()
            if cs in {"year", "年份", "骞翠唤"}:
                df = df.rename(columns={c: YEAR_COL})
                break

    if ID_COL not in df.columns:
        df[ID_COL] = np.arange(len(df)).astype(str)
        log_step(f"[WARN] 缺少 {ID_COL}，已自动生成临时ID。")
    if YEAR_COL not in df.columns:
        df[YEAR_COL] = np.arange(len(df), dtype="int64")
        log_step(f"[WARN] 缺少 {YEAR_COL}，已自动生成临时年份序列。")

    available_y = [c for c in Y_COLS if c in df.columns]
    if not available_y:
        raise ValueError(f"目标变量全部缺失，无法运行: {Y_COLS}")
    if len(available_y) < len(Y_COLS):
        log_step(f"[WARN] 部分目标变量缺失，已跳过: {[c for c in Y_COLS if c not in available_y]}")
    Y_cols_runtime = available_y

    all_x_raw = sorted(set(sum(FEATURE_SETS_RAW.values(), [])))
    missing_x = [c for c in all_x_raw if c not in df.columns]
    available_x = [c for c in all_x_raw if c in df.columns]
    if missing_x:
        log_step(f"[WARN] 以下特征列不存在，已自动跳过: {missing_x}")
    if not available_x:
        raise ValueError("没有可用特征列，无法建模。")

    feature_sets_raw = {}
    for fs_name in FEAT_ORDER:
        cols = FEATURE_SETS_RAW[fs_name]
        keep_cols = [c for c in cols if c in df.columns]
        if len(keep_cols) == 0:
            log_step(f"[WARN] 特征集 {fs_name} 无可用列，将不参与实验。")
            continue
        feature_sets_raw[fs_name] = keep_cols

    if "X_CONTROLS" not in feature_sets_raw:
        raise ValueError("X_CONTROLS 全部缺失，无法进行基线比较。")

    models = build_models()
    log_step(f"[INFO] 模型列表: {list(models.keys())}")

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, YEAR_COL]).copy()
    df[ID_COL] = df[ID_COL].astype(str)

    for c in available_x + Y_cols_runtime:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    lag_suffix = f"_lag{LAG_K}"
    for c in available_x:
        df[f"{c}{lag_suffix}"] = df.groupby(ID_COL, sort=False)[c].shift(LAG_K)

    feature_sets = {k: [f"{c}{lag_suffix}" for c in v] for k, v in feature_sets_raw.items()}
    all_x_lag = sorted(set(sum(feature_sets.values(), [])))

    for c in all_x_lag + Y_cols_runtime:
        df[c] = winsorize(df[c], WINSOR_P)

    if USE_LN1P_Y:
        for y in Y_cols_runtime:
            bad = df[y] < -1
            if bad.any():
                df.loc[bad, y] = np.nan
            df[y] = np.log1p(df[y])

    df = df.dropna(subset=all_x_lag).copy()
    log_step(f"[INFO] after lag+winsor+filter: n={len(df)}")
    log_step("[INFO] 开始执行实验：city + roll")

    for exp in ["city", "roll"]:
        log_step(f"[INFO] 进入实验阶段: {exp}")
        fold_df, summary_df, tests_df = run_experiment(exp, df, models, feature_sets)

        fold_path = os.path.join(log_dir, f"{exp}_fold_metrics_lag1.csv")
        summ_path = os.path.join(log_dir, f"{exp}_summary_lag1.csv")
        test_path = os.path.join(log_dir, f"{exp}_tests_with_multipletests_lag1.csv")

        fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summ_path, index=False, encoding="utf-8-sig")
        tests_df.to_csv(test_path, index=False, encoding="utf-8-sig")
        if exp == "city":
            export_city_plots_and_tables(summary_df, tests_df, log_dir)
            log_step("[INFO] city实验图表和检验表导出完成")

        log_step(f"================ {exp.upper()} DONE (X lag1 -> y_t) ================")
        log_step(f"[OK] fold   -> {fold_path}")
        log_step(f"[OK] summary-> {summ_path}")
        log_step(f"[OK] tests  -> {test_path}")

        if not tests_df.empty:
            show_cols = [
                "exp",
                "model",
                "y",
                "compare",
                "n_splits",
                "mean_dR2",
                "p_t_R2_holm",
                "p_w_R2_holm",
                "mean_dMAE",
                "p_t_MAE_holm",
                "p_w_MAE_holm",
            ]
            log_step("[INFO] Holm-adjusted key results 如下")
            print(tests_df[show_cols].to_string(index=False), flush=True)
        else:
            log_step("[WARN] tests 为空：Rolling 可能因年份跨度/窗口/缺失导致没有可用 split。")

    log_step("ALL DONE.")


if __name__ == "__main__":
    main()
