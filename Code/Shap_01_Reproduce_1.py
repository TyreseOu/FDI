# -*- coding: utf-8 -*-

import os
import warnings

warnings.filterwarnings("ignore")
import logging

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

import xgboost
from xgboost import XGBRegressor

try:
    import shap

    _HAS_SHAP = True
except Exception as e:
    _HAS_SHAP = False
    _SHAP_ERR = e

try:
    from scipy.stats import gaussian_kde

    _HAS_KDE = True
except Exception:
    _HAS_KDE = False

installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_candidates = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
selected_fonts = [f for f in font_candidates if f in installed_fonts]
if not selected_fonts:
    selected_fonts = ["DejaVu Sans"]
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    print("[WARN] 未检测到中文字体，将回退 DejaVu Sans（中文可能显示为方框）。")

rcParams["font.sans-serif"] = selected_fonts
rcParams["axes.unicode_minus"] = False
print(f"[INFO] 当前绘图字体: {selected_fonts}")

PATH = r"Dataset.csv"
LOG_DIR = r"shap_log_01"
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except PermissionError:
    LOG_DIR = os.path.abspath("./shap_log_01")
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[WARN] 无权限写入默认输出目录，已回退到: {LOG_DIR}")

ID_COL = "市级行政代码"
YEAR_COL = "年份"

N_SPLITS = 5
SEED = 21

USE_LN1P_Y = True
WINSOR_P = 0.01
LAG_K = 1

DO_SHAP = True
SHAP_MAX_N = 3000
SHAP_RANDOM_SEED = 123
SHAP_PLOT = True
TOPK_PRINT = 50
SHAP_BAR_PLOT = True

DO_DEPENDENCE = True
DEP_TOPK = 6

DENSITY_SCATTER = True
DENSITY_METHOD = "kde"
DENSITY_GRIDSIZE = 60
DENSITY_POINT_SIZE = 12
DENSITY_ALPHA = 0.95

DO_YEAR_IMPORTANCE = True
YEAR_TOPK = 10
SAVE_YEAR_IMPORTANCE = True
SAVE_PLOTS_PNG = True
PLOT_DIR = os.path.join(LOG_DIR, "figs")
if SAVE_PLOTS_PNG:
    os.makedirs(PLOT_DIR, exist_ok=True)

Y_COLS = [
    "外商直接投资（百万美元）",
    "外商直接投资额（百万元人民币）",
]

X_CORE = [
    "案件总数", "案件数占人口比例", "审理费占人口比例", "上诉率", "再审率", "裁判文书说理平均字数",
    "一审上诉改判率", "一审上诉发回重审率", "生效案件改判率", "案件发回重审率",
    "一审陪审率", "二审开庭率", "独任审理率", "人均律师数量", "案件平均审理天数",
]

X_CORE_D = [
    "案件总数", "案件数占人口比例", "审理费占人口比例", "上诉率", "再审率", "裁判文书说理平均字数",
    "一审上诉改判率", "一审上诉发回重审率", "生效案件改判率", "案件发回重审率",
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

X_ALL_RAW = X_CONTROLS + X_CORE

MODEL_NAME = "XGB"
MODEL = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1
)


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
    return Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("m", clone(model_template)),
    ])


def density_scatter_xy(ax, x, y, method="kde", gridsize=60,
                       s=12, alpha=0.95, cmap="viridis"):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m];
    y = y[m]
    if len(x) == 0:
        ax.text(0.5, 0.5, "no valid points", ha="center", va="center", transform=ax.transAxes)
        return None

    if method == "kde" and _HAS_KDE and len(x) >= 30:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = np.argsort(z)
        x2, y2, z2 = x[idx], y[idx], z[idx]
        sc = ax.scatter(x2, y2, c=z2, s=s, alpha=alpha, cmap=cmap, edgecolors="none")
        return sc
    else:
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1)
        return hb


print(f"[INFO] 读取数据: {PATH}")
print(f"[INFO] 输出目录: {LOG_DIR}")
df = pd.read_csv(PATH, encoding="utf-8-sig")

rename_map = {}
if "外商直接投资额（百万人民币）" in df.columns and "外商直接投资额（百万元人民币）" not in df.columns:
    rename_map["外商直接投资额（百万人民币）"] = "外商直接投资额（百万元人民币）"
if "外商直接投资额(百万人民币)" in df.columns and "外商直接投资额（百万元人民币）" not in df.columns:
    rename_map["外商直接投资额(百万人民币)"] = "外商直接投资额（百万元人民币）"
if "标的额占人口比例" in df.columns and "审理费占人口比例" not in df.columns:
    rename_map["标的额占人口比例"] = "审理费占人口比例"
if "中等职业教育学校在校学生数(人)" in df.columns and "中等职业教育学校在校学生数（人）全市" not in df.columns:
    rename_map["中等职业教育学校在校学生数(人)"] = "中等职业教育学校在校学生数（人）全市"
if "地区生产总值增长率(%)" in df.columns and "地区生产总值增长率-全市(%)" not in df.columns:
    rename_map["地区生产总值增长率(%)"] = "地区生产总值增长率-全市(%)"
if "地方财政一般预算内收入(万元)" in df.columns and "地方一般公共预算收入（万元）全市" not in df.columns:
    rename_map["地方财政一般预算内收入(万元)"] = "地方一般公共预算收入（万元）全市"
if "小学专任教师数(人)" in df.columns and "普通小学专任教师数（人）全市" not in df.columns:
    rename_map["小学专任教师数(人)"] = "普通小学专任教师数（人）全市"
if "工业二氧化硫排放量(吨)" in df.columns and "工业二氧化硫排放量（吨）全市" not in df.columns:
    rename_map["工业二氧化硫排放量(吨)"] = "工业二氧化硫排放量（吨）全市"
if "普通中学学校数(所)" in df.columns and "普通中学（所）全市" not in df.columns:
    rename_map["普通中学学校数(所)"] = "普通中学（所）全市"
if "普通高等学校专任教师数(人)" in df.columns and "普通高等学校专任教师数（人）全市" not in df.columns:
    rename_map["普通高等学校专任教师数(人)"] = "普通高等学校专任教师数（人）全市"
if "污水处理厂集中处理率(%)" in df.columns and "污水处理厂集中处理率（%）全市" not in df.columns:
    rename_map["污水处理厂集中处理率(%)"] = "污水处理厂集中处理率（%）全市"
if "电信业务收入(万元)" in df.columns and "电信业务收入（万元）全市" not in df.columns:
    rename_map["电信业务收入(万元)"] = "电信业务收入（万元）全市"
if "职工基本医疗保险参保人数(人)" in df.columns and "城镇职工基本医疗保险参保人数（人）全市" not in df.columns:
    rename_map["职工基本医疗保险参保人数(人)"] = "城镇职工基本医疗保险参保人数（人）全市"
if "规模以上工业企业利润总额(万元)" in df.columns and "规模以上工业企业主要财务指标_利润总额（万元）全市" not in df.columns:
    rename_map["规模以上工业企业利润总额(万元)"] = "规模以上工业企业主要财务指标_利润总额（万元）全市"
if "规模以上工业企业数(个)" in df.columns and "规模以上工业企业数（个）全市" not in df.columns:
    rename_map["规模以上工业企业数(个)"] = "规模以上工业企业数（个）全市"
if "邮政业务收入(万元)" in df.columns and "邮政业务收入（万元）全市" not in df.columns:
    rename_map["邮政业务收入(万元)"] = "邮政业务收入（万元）全市"
if "限额以上批发零售业商品销售总额(万元)" in df.columns and "限额以上批发零售业商品销售总额（万元）全市" not in df.columns:
    rename_map["限额以上批发零售业商品销售总额(万元)"] = "限额以上批发零售业商品销售总额（万元）全市"
if "限额以上批发零售业法人企业数(个)" in df.columns and "限额以上批发零售商贸企业数_法人数（个）全市" not in df.columns:
    rename_map["限额以上批发零售业法人企业数(个)"] = "限额以上批发零售商贸企业数_法人数（个）全市"
if rename_map:
    df = df.rename(columns=rename_map)

required_cols = [ID_COL, YEAR_COL] + Y_COLS
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"必要列不存在：{missing_required}")

missing_controls = [c for c in X_CONTROLS if c not in df.columns]
missing_core = [c for c in X_CORE if c not in df.columns]
if missing_controls:
    print(f"[WARN] 下列控制变量缺失，已跳过: {missing_controls}")
if missing_core:
    print(f"[WARN] 下列核心变量缺失，已跳过: {missing_core}")

X_CONTROLS = [c for c in X_CONTROLS if c in df.columns]
X_CORE = [c for c in X_CORE if c in df.columns]
X_CORE_D = [c for c in X_CORE_D if c in df.columns]
X_CORE_P = [c for c in X_CORE_P if c in df.columns]
X_ALL_RAW = X_CONTROLS + X_CORE

if len(X_CONTROLS) == 0 or len(X_CORE) == 0:
    raise ValueError("可用控制变量或核心变量为空，无法继续SHAP流程。")

df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df = df.dropna(subset=[ID_COL, YEAR_COL]).copy()
df[ID_COL] = df[ID_COL].astype(str)

for c in X_ALL_RAW + Y_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)
LAG_SUFFIX = f"_lag{LAG_K}"
for c in X_ALL_RAW:
    df[c + LAG_SUFFIX] = df.groupby(ID_COL, sort=False)[c].shift(LAG_K)

X_ALL = [c + LAG_SUFFIX for c in X_ALL_RAW]

for c in X_ALL + Y_COLS:
    df[c] = winsorize(df[c], WINSOR_P)

if USE_LN1P_Y:
    for y in Y_COLS:
        bad = df[y] < -1
        if bad.any():
            df.loc[bad, y] = np.nan
        df[y] = np.log1p(df[y].clip(lower=0))

df = df.dropna(subset=X_ALL).copy()
print(f"[INFO] after lag+winsor+filter: n={len(df)}")

fold_rows = []
summary_rows = []

gkf = GroupKFold(n_splits=N_SPLITS)

for y in Y_COLS:
    dfy = df.dropna(subset=[y]).copy().reset_index(drop=True)
    if dfy.empty:
        continue

    yv = dfy[y].values
    groups = dfy[ID_COL].astype(str).values
    X = dfy[X_ALL]

    r2_list, mae_list = [], []

    for fold_id, (tr, te) in enumerate(gkf.split(np.zeros(len(dfy)), groups=groups)):
        pipe = make_pipe(MODEL)
        pipe.fit(X.iloc[tr], yv[tr])
        pred = pipe.predict(X.iloc[te])

        r2 = oos_r2(yv[te], pred)
        mae = mean_absolute_error(yv[te], pred)

        r2_list.append(r2)
        mae_list.append(mae)

        fold_rows.append({
            "exp": "city",
            "model": MODEL_NAME,
            "y": y,
            "feature_set": "ALL(X_CONTROLS+X_CORE)",
            "split_id": int(fold_id),
            "fold": int(fold_id),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "MAE": float(mae) if np.isfinite(mae) else np.nan,
            "n_test": int(len(te)),
        })

    r2_arr = np.asarray(r2_list, dtype=float)
    mae_arr = np.asarray(mae_list, dtype=float)

    summary_rows.append({
        "exp": "city",
        "model": MODEL_NAME,
        "y": y,
        "feature_set": "ALL(X_CONTROLS+X_CORE)",
        "mean_R2": float(np.nanmean(r2_arr)),
        "median_R2": float(np.nanmedian(r2_arr)),
        "std_R2": float(np.nanstd(r2_arr, ddof=0)),
        "mean_MAE": float(np.nanmean(mae_arr)),
        "median_MAE": float(np.nanmedian(mae_arr)),
        "std_MAE": float(np.nanstd(mae_arr, ddof=0)),
        "N_eval": int(np.isfinite(r2_arr).sum()),
    })

fold_df = pd.DataFrame(fold_rows)
summary_df = pd.DataFrame(summary_rows).sort_values(["y"]).reset_index(drop=True)

fold_path = os.path.join(LOG_DIR, "city_xgb_allfeatures_fold_metrics_lag1.csv")
summ_path = os.path.join(LOG_DIR, "city_xgb_allfeatures_summary_lag1.csv")
fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summ_path, index=False, encoding="utf-8-sig")

print("\n================ CITY DONE (X lag1 -> y_t) ================")
print(f"[OK] fold   -> {fold_path}")
print(f"[OK] summary-> {summ_path}")
print("\n[Summary]")
print(summary_df.to_string(index=False))

if DO_SHAP:
    rng = np.random.RandomState(SHAP_RANDOM_SEED)
    print("\n================ SHAP ANALYSIS (XGB native pred_contribs) ================")

    FEATS_CTRL = [c + LAG_SUFFIX for c in X_CONTROLS]
    FEATS_CORE_D = [c + LAG_SUFFIX for c in X_CORE_D]
    FEATS_CORE_P = [c + LAG_SUFFIX for c in X_CORE_P]
    FEATS_CORE_ALL = FEATS_CORE_D + FEATS_CORE_P

    COLOR_MAP = {
        "X_CONTROLS": "#4C78A8",
        "X_CORE_D": "#F58518",
        "X_CORE_P": "#54A24B",
    }

    for y in Y_COLS:
        dfy = df.dropna(subset=[y]).copy().reset_index(drop=True)
        if dfy.empty:
            continue

        X_raw = dfy[X_ALL]
        yv = dfy[y].values
        feature_names = list(X_raw.columns)

        imp = SimpleImputer(strategy="mean")
        X_imp = imp.fit_transform(X_raw)

        n = X_imp.shape[0]
        if n > SHAP_MAX_N:
            idx_used = rng.choice(n, size=SHAP_MAX_N, replace=False)
            X_used = X_imp[idx_used]
        else:
            idx_used = np.arange(n)
            X_used = X_imp

        model_full = clone(MODEL)
        model_full.fit(X_imp, yv)

        dmat = xgboost.DMatrix(X_used, feature_names=feature_names)
        contrib = model_full.get_booster().predict(dmat, pred_contribs=True)
        shap_vals = contrib[:, :-1]

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})

        y_short = "FDI(美元)" if "美元" in y else "FDI(人民币)"
        y_safe = "usd" if "美元" in y else "cny"
        print(f"\n[SHAP Top {TOPK_PRINT} | overall] y = {y}")
        print(imp_df.sort_values("mean_abs_shap", ascending=False).head(TOPK_PRINT).to_string(index=False))

        imp_all_path = os.path.join(LOG_DIR, f"shap_importance_all_{MODEL_NAME}_{y_short}_lag1.csv")
        imp_topk_path = os.path.join(LOG_DIR, f"shap_importance_top{TOPK_PRINT}_{MODEL_NAME}_{y_short}_lag1.csv")
        imp_df.sort_values("mean_abs_shap", ascending=False).to_csv(imp_all_path, index=False, encoding="utf-8-sig")
        imp_df.sort_values("mean_abs_shap", ascending=False).head(TOPK_PRINT).to_csv(
            imp_topk_path, index=False, encoding="utf-8-sig"
        )
        print(f"[OK] SHAP重要性全表 -> {imp_all_path}")
        print(f"[OK] SHAP重要性TopK -> {imp_topk_path}")

        if SHAP_PLOT:
            if not _HAS_SHAP:
                print("\n[WARN] shap 未安装，无法画 summary_plot。")
                print(f"原因：{repr(_SHAP_ERR)}")
            else:
                try:
                    ordered_features = [f for f in FEATS_CORE_ALL if f in feature_names]
                    if len(ordered_features) == 0:
                        print("[WARN] X_CORE 特征在 feature_names 中未找到，跳过 summary_plot。")
                    else:
                        col_idx = [feature_names.index(f) for f in ordered_features]
                        X_plot = pd.DataFrame(X_used, columns=feature_names)[ordered_features]
                        shap_plot_vals = shap_vals[:, col_idx]

                        plt.figure(figsize=(7.5, 5.2))
                        shap.summary_plot(shap_plot_vals, X_plot, show=False)
                        plt.title(f"SHAP 汇总图（仅 X_CORE）｜{y_short}", fontsize=12)
                        plt.tight_layout()
                        if SAVE_PLOTS_PNG:
                            out_png = os.path.join(PLOT_DIR, f"shap_summary_core_{MODEL_NAME}_{y_safe}_lag1.png")
                            plt.savefig(out_png, dpi=300, bbox_inches="tight")
                            print(f"[OK] SHAP汇总图 -> {out_png}")
                        plt.close()
                except Exception as e:
                    print(f"[WARN] summary_plot 失败（多半是 matplotlib 环境问题）：{repr(e)}")

        if SHAP_BAR_PLOT:
            try:
                controls_df = imp_df[imp_df["feature"].isin(FEATS_CTRL)].copy().sort_values("mean_abs_shap",
                                                                                            ascending=False)
                coreD_df = imp_df[imp_df["feature"].isin(FEATS_CORE_D)].copy().sort_values("mean_abs_shap",
                                                                                           ascending=False)
                coreP_df = imp_df[imp_df["feature"].isin(FEATS_CORE_P)].copy().sort_values("mean_abs_shap",
                                                                                           ascending=False)

                plot_df = pd.concat([controls_df, coreD_df, coreP_df], axis=0).reset_index(drop=True)
                plot_df = plot_df.head(TOPK_PRINT).copy()

                set_ctrl = set(FEATS_CTRL)
                set_d = set(FEATS_CORE_D)
                set_p = set(FEATS_CORE_P)


                def _which_group(f):
                    if f in set_ctrl: return "X_CONTROLS"
                    if f in set_d:    return "X_CORE_D"
                    if f in set_p:    return "X_CORE_P"
                    return "OTHER"


                plot_df["group"] = plot_df["feature"].map(_which_group)
                plot_df["color"] = plot_df["group"].map(COLOR_MAP).fillna("#888888")

                fig, ax = plt.subplots(figsize=(7.2, 6.0))
                ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=plot_df["color"])
                ax.invert_yaxis()
                ax.set_xlabel("mean(|SHAP|)")
                ax.set_title(
                    f"特征重要性（mean|SHAP|）｜{y_short}\n"
                    f"（组顺序：X_CONTROLS → X_CORE_D → X_CORE_P；组内降序）"
                )

                ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                from matplotlib.patches import Patch

                legend_handles = [
                    Patch(color=COLOR_MAP["X_CONTROLS"], label="X_CONTROLS"),
                    Patch(color=COLOR_MAP["X_CORE_D"], label="X_CORE_D（分配正义）"),
                    Patch(color=COLOR_MAP["X_CORE_P"], label="X_CORE_P（程序正义）"),
                ]
                ax.legend(handles=legend_handles, loc="lower right", frameon=False)

                plt.tight_layout()
                if SAVE_PLOTS_PNG:
                    out_png = os.path.join(PLOT_DIR, f"shap_bar_grouped_{MODEL_NAME}_{y_safe}_lag1.png")
                    plt.savefig(out_png, dpi=300, bbox_inches="tight")
                    print(f"[OK] SHAP分组条形图 -> {out_png}")
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] 重要性条形图绘制失败：{repr(e)}")

        if DO_DEPENDENCE:
            try:
                X_used_df = pd.DataFrame(X_used, columns=feature_names)

                core_feats = [f for f in FEATS_CORE_ALL if f in feature_names]
                if len(core_feats) == 0:
                    print("[WARN] X_CORE 特征未找到，跳过 dependence plot。")
                else:
                    mean_abs_s = pd.Series(mean_abs, index=feature_names)
                    core_rank = mean_abs_s.loc[core_feats].sort_values(ascending=False)
                    dep_feats = core_rank.index[:min(DEP_TOPK, len(core_rank))].tolist()

                    print(f"\n[Dependence-density] y={y_short} | X_CORE Top{len(dep_feats)}:")
                    for f in dep_feats:
                        print(" -", f)

                    method = DENSITY_METHOD.lower()
                    if method == "kde" and not _HAS_KDE:
                        print("[WARN] scipy 不可用，kde 密度着色不可用，已自动改用 hexbin。")
                        method = "hexbin"

                    for f in dep_feats:
                        x_feat = X_used_df[f].values
                        j = feature_names.index(f)
                        y_shap = shap_vals[:, j]

                        fig, ax = plt.subplots(figsize=(6.4, 4.8))

                        if DENSITY_SCATTER:
                            mappable = density_scatter_xy(
                                ax=ax,
                                x=x_feat,
                                y=y_shap,
                                method=method,
                                gridsize=DENSITY_GRIDSIZE,
                                s=DENSITY_POINT_SIZE,
                                alpha=DENSITY_ALPHA,
                                cmap="viridis",
                            )
                            if mappable is not None:
                                cb = fig.colorbar(mappable, ax=ax)
                                cb.set_label("点密度（越深越密）")
                        else:
                            ax.scatter(x_feat, y_shap, s=10, alpha=0.75)

                        ax.axhline(0.0, linewidth=0.8, linestyle="--", alpha=0.6)
                        ax.set_xlabel(f"{f}")
                        ax.set_ylabel("SHAP value")
                        ax.set_title(f"Dependence（密度着色）｜{y_short}\n{f}", fontsize=12)
                        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

                        plt.tight_layout()
                        if SAVE_PLOTS_PNG:
                            f_safe = str(f).replace("/", "_").replace("\\", "_")
                            out_png = os.path.join(PLOT_DIR, f"shap_dependence_{MODEL_NAME}_{y_safe}_{f_safe}_lag1.png")
                            plt.savefig(out_png, dpi=300, bbox_inches="tight")
                            print(f"[OK] Dependence图 -> {out_png}")
                        plt.close(fig)
            except Exception as e:
                print(f"[WARN] dependence（密度）绘制失败：{repr(e)}")

        if DO_YEAR_IMPORTANCE:
            try:
                year_used = pd.to_numeric(dfy.loc[idx_used, YEAR_COL], errors="coerce").values
                ok = np.isfinite(year_used)
                year_used = year_used[ok].astype(int)
                shap_ok = shap_vals[ok, :]

                core_feats = [f for f in FEATS_CORE_ALL if f in feature_names]
                if len(core_feats) == 0:
                    print("[WARN] X_CORE 特征未找到，跳过年份分层重要性。")
                else:
                    core_idx = [feature_names.index(f) for f in core_feats]

                    tmp = []
                    for yr in np.sort(np.unique(year_used)):
                        m = year_used == yr
                        mean_abs_y = np.mean(np.abs(shap_ok[m][:, core_idx]), axis=0)
                        tmp.append(pd.DataFrame({
                            "year": yr,
                            "feature": core_feats,
                            "mean_abs_shap": mean_abs_y
                        }))
                    year_imp_long = pd.concat(tmp, ignore_index=True)

                    if SAVE_YEAR_IMPORTANCE:
                        out_long = os.path.join(LOG_DIR, f"shap_year_meanabs_CORE_{MODEL_NAME}_{y_short}_lag1_long.csv")
                        year_imp_long.to_csv(out_long, index=False, encoding="utf-8-sig")
                        out_wide = os.path.join(LOG_DIR, f"shap_year_meanabs_CORE_{MODEL_NAME}_{y_short}_lag1_wide.csv")
                        year_imp_long.pivot_table(index="year", columns="feature", values="mean_abs_shap").to_csv(
                            out_wide, encoding="utf-8-sig")
                        print(f"[OK] 年份分层重要性（long）-> {out_long}")
                        print(f"[OK] 年份分层重要性（wide）-> {out_wide}")

                    core_avg = (year_imp_long.groupby("feature")["mean_abs_shap"]
                                .mean().sort_values(ascending=False))
                    top_feats = core_avg.index[:min(YEAR_TOPK, len(core_avg))].tolist()

                    wide = year_imp_long.pivot_table(index="year", columns="feature",
                                                     values="mean_abs_shap").sort_index()
                    plt.figure(figsize=(7.6, 4.8))
                    for f in top_feats:
                        if f in wide.columns:
                            plt.plot(wide.index.values, wide[f].values, marker="o", linewidth=1.5, label=f)
                    plt.title(f"年份分层 mean(|SHAP|) 漂移（仅 X_CORE Top{len(top_feats)}）｜{y_short}", fontsize=12)
                    plt.xlabel("年份")
                    plt.ylabel("mean(|SHAP|)")
                    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                    plt.legend(loc="best", fontsize=8, frameon=False)
                    plt.tight_layout()
                    if SAVE_PLOTS_PNG:
                        out_png = os.path.join(PLOT_DIR, f"shap_year_trend_core_{MODEL_NAME}_{y_safe}_lag1.png")
                        plt.savefig(out_png, dpi=300, bbox_inches="tight")
                        print(f"[OK] 年份漂移图 -> {out_png}")
                    plt.close()
            except Exception as e:
                print(f"[WARN] 年份分层重要性绘制失败：{repr(e)}")

print("\nALL DONE.")
