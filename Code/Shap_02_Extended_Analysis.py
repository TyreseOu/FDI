#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

from xgboost import XGBRegressor
import xgboost

try:
    import shap
except Exception as e:
    raise ImportError(f"需要先安装 shap：pip install shap。当前错误: {repr(e)}")

DEFAULT_PATH = "Dataset.csv"
DEFAULT_OUT_DIR = "shap_log_02_extended"

ID_COL = "市级行政代码"
YEAR_COL = "年份"
PROV_COL = "省份"
CITY_COL = "城市"

Y_COLS = [
    "外商直接投资（百万美元）",
    "外商直接投资额（百万元人民币）",
]

X_CORE = [
    "案件总数", "案件数占人口比例", "审理费占人口比例", "上诉率", "再审率", "裁判文书说理平均字数",
    "一审上诉改判率", "一审上诉发回重审率", "生效案件改判率", "案件发回重审率",
    "一审陪审率", "二审开庭率", "独任审理率", "人均律师数量", "案件平均审理天数",
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

LAG_K = 1
WINSOR_P = 0.01
USE_LN1P_Y = True
LOCAL_TOP_K = 5
INTERACTION_SAMPLE_N = 0
INTERACTION_TOP_K = 50
MONITOR_TOP_K = 8


def log_step(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def set_chinese_font() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "Source Han Sans CN",
        "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "Arial Unicode MS",
    ]
    chosen = [f for f in candidates if f in installed]
    if not chosen:
        chosen = ["DejaVu Sans"]
    rcParams["font.sans-serif"] = chosen
    rcParams["axes.unicode_minus"] = False
    log_step(f"[INFO] 当前字体: {chosen}")


def read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding_errors="replace")


def normalize_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    aliases = {
        "外商直接投资额（百万人民币）": "外商直接投资额（百万元人民币）",
        "外商直接投资额(百万人民币)": "外商直接投资额（百万元人民币）",
        "标的额占人口比例": "审理费占人口比例",
        "城镇非私营单位在岗职工平均工资(元)": "城镇非私营单位在岗职工平均工资（元）",
        "甯傜骇琛屾斂浠ｇ爜": "市级行政代码",
        "骞翠唤": "年份",
    }
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def winsorize(s: pd.Series, p: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def east_central_west(province: str) -> str:
    east = {"北京市", "天津市", "河北省", "上海市", "江苏省", "浙江省", "福建省", "山东省", "广东省", "海南省"}
    central = {"山西省", "安徽省", "江西省", "河南省", "湖北省", "湖南省"}
    west = {"内蒙古自治区", "广西壮族自治区", "重庆市", "四川省", "贵州省", "云南省", "西藏自治区", "陕西省", "甘肃省",
            "青海省", "宁夏回族自治区", "新疆维吾尔自治区"}
    northeast = {"辽宁省", "吉林省", "黑龙江省"}
    p = str(province).strip()
    if p in east:
        return "东部"
    if p in central:
        return "中部"
    if p in west:
        return "西部"
    if p in northeast:
        return "东北"
    return "其他"


def ensure_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL not in df.columns:
        df[ID_COL] = np.arange(len(df)).astype(str)
        log_step(f"[WARN] 缺少 {ID_COL}，已生成临时ID")
    if YEAR_COL not in df.columns:
        df[YEAR_COL] = np.arange(len(df), dtype="int64")
        log_step(f"[WARN] 缺少 {YEAR_COL}，已生成临时年份")
    if PROV_COL not in df.columns:
        df[PROV_COL] = "未知省份"
        log_step(f"[WARN] 缺少 {PROV_COL}，将仅使用'未知省份'")
    if CITY_COL not in df.columns:
        df[CITY_COL] = "未知城市"
        log_step(f"[WARN] 缺少 {CITY_COL}，将仅使用'未知城市'")
    return df


def prepare_data(path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = read_csv_smart(path)
    df = normalize_aliases(df)
    df = ensure_required_cols(df)

    y_cols = [c for c in Y_COLS if c in df.columns]
    if not y_cols:
        raise ValueError(f"目标列均不存在: {Y_COLS}")
    if len(y_cols) < len(Y_COLS):
        log_step(f"[WARN] 部分目标列缺失，已跳过: {[c for c in Y_COLS if c not in y_cols]}")

    x_raw = [c for c in (X_CONTROLS + X_CORE) if c in df.columns]
    miss = [c for c in (X_CONTROLS + X_CORE) if c not in df.columns]
    if miss:
        log_step(f"[WARN] 以下特征缺失，已跳过: {miss}")
    if not x_raw:
        raise ValueError("可用特征为空，无法运行 SHAP 扩展分析。")

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[ID_COL] = df[ID_COL].astype(str)
    df = df.dropna(subset=[ID_COL, YEAR_COL]).copy()

    for c in x_raw + y_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    lag_cols = []
    for c in x_raw:
        lag_col = f"{c}_lag{LAG_K}"
        df[lag_col] = df.groupby(ID_COL, sort=False)[c].shift(LAG_K)
        lag_cols.append(lag_col)

    for c in lag_cols + y_cols:
        df[c] = winsorize(df[c], WINSOR_P)

    if USE_LN1P_Y:
        for y in y_cols:
            df[y] = np.log1p(df[y].clip(lower=0))

    df = df.dropna(subset=lag_cols).copy()
    log_step(f"[INFO] after lag+winsor+filter: n={len(df)}")
    return df, lag_cols, y_cols


def fit_xgb(df: pd.DataFrame, x_cols: List[str], y_col: str):
    tmp = df.dropna(subset=[y_col]).copy()
    X = tmp[x_cols]
    y = tmp[y_col].values
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=21,
        n_jobs=-1,
    )
    model.fit(X, y)
    pred = model.predict(X)
    tmp = tmp.reset_index(drop=True)

    explainer = None
    exp = None
    try:
        explainer = shap.TreeExplainer(model)
        exp = explainer(X)
    except Exception as e:
        log_step(f"[WARN] TreeExplainer失败，回退到XGBoost原生pred_contribs: {e}")
        booster = model.get_booster()
        dmat = xgboost.DMatrix(X, feature_names=x_cols)
        contrib = booster.predict(dmat, pred_contribs=True, validate_features=False)
        exp = shap.Explanation(
            values=contrib[:, :-1],
            base_values=contrib[:, -1],
            data=X.values,
            feature_names=x_cols,
        )
    return model, explainer, exp, pred, tmp


def select_representative_indices(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 5) -> List[
    int]:
    n = len(df)
    if n == 0:
        return []
    resid = y_true - y_pred
    idxs = set()
    idxs.add(int(np.argmax(y_pred)))
    idxs.add(int(np.argmin(y_pred)))
    idxs.add(int(np.argmin(np.abs(y_pred - np.median(y_pred)))))
    idxs.add(int(np.argmax(resid)))
    idxs.add(int(np.argmin(resid)))
    return list(sorted(idxs))[:top_k]


def run_local_waterfall(exp, df_used: pd.DataFrame, y_col: str, y_pred: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    y_true = df_used[y_col].values
    idxs = select_representative_indices(df_used, y_true, y_pred, top_k=LOCAL_TOP_K)
    rows = []
    for idx in idxs:
        city = str(df_used.iloc[idx][CITY_COL])
        year = str(df_used.iloc[idx][YEAR_COL])
        tag = f"idx{idx}_{city}_{year}".replace("/", "_")
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp[idx], max_display=20, show=False)
        plt.tight_layout()
        png = os.path.join(out_dir, f"waterfall_{y_col}_{tag}.png")
        plt.savefig(png, dpi=180, bbox_inches="tight")
        plt.close()

        contrib = pd.DataFrame({
            "feature": exp.feature_names,
            "shap_value": exp.values[idx],
            "abs_shap": np.abs(exp.values[idx]),
        }).sort_values("abs_shap", ascending=False)
        csv = os.path.join(out_dir, f"waterfall_contrib_{y_col}_{tag}.csv")
        contrib.to_csv(csv, index=False, encoding="utf-8-sig")

        rows.append({
            "idx": idx, "城市": city, "年份": year,
            "y_true": y_true[idx], "y_pred": y_pred[idx], "residual": y_true[idx] - y_pred[idx],
            "png": png, "csv": csv,
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"waterfall_cases_{y_col}.csv"), index=False, encoding="utf-8-sig")
    log_step(f"[OK] 局部waterfall完成: {y_col} | 样本数={len(rows)}")


def run_cohorts_and_group_diff(exp, df_used: pd.DataFrame, y_col: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    region = df_used[PROV_COL].map(east_central_west)
    fdi_group = np.where(df_used[y_col] >= np.nanmedian(df_used[y_col].values), "高FDI", "低FDI")

    grp_defs = {
        "region_group": pd.Series(region, index=df_used.index),
        "fdi_group": pd.Series(fdi_group, index=df_used.index),
    }

    for gname, gser in grp_defs.items():
        out_rows = []
        for g in sorted(gser.dropna().unique()):
            m = (gser == g).values
            if m.sum() == 0:
                continue
            mean_abs = np.abs(exp.values[m]).mean(axis=0)
            part = pd.DataFrame({
                "group_name": gname,
                "group": g,
                "feature": exp.feature_names,
                "mean_abs_shap": mean_abs,
            }).sort_values("mean_abs_shap", ascending=False)
            out_rows.append(part)
        if out_rows:
            out_df = pd.concat(out_rows, ignore_index=True)
            out_df.to_csv(os.path.join(out_dir, f"cohorts_meanabs_{y_col}_{gname}.csv"), index=False,
                          encoding="utf-8-sig")

        try:
            cohorts = exp.cohorts(gser.astype(str).values)
            plt.figure(figsize=(10, 7))
            shap.plots.bar(cohorts, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cohorts_bar_{y_col}_{gname}.png"), dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            log_step(f"[WARN] cohorts bar失败({gname}, {y_col}): {e}")

        if gname == "fdi_group":
            mask = (gser.values == "高FDI")
            ok = False
            for call in (
                    lambda: shap.plots.group_difference(exp.values, mask, feature_names=exp.feature_names, show=False),
                    lambda: shap.plots.group_difference(exp, mask, show=False),
                    lambda: shap.plots.group_difference(mask, exp, show=False),
            ):
                try:
                    plt.figure(figsize=(10, 7))
                    call()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"group_difference_{y_col}_{gname}.png"), dpi=180,
                                bbox_inches="tight")
                    plt.close()
                    ok = True
                    break
                except Exception:
                    plt.close()
            if not ok:
                log_step(f"[WARN] group_difference绘图失败({gname}, {y_col})，已保留CSV统计结果")

    log_step(f"[OK] 分组差异SHAP完成: {y_col}")


def run_interaction_shap(model, explainer, exp, df_used: pd.DataFrame, y_col: str, out_dir: str,
                         interaction_sample_n: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    X = pd.DataFrame(exp.data, columns=exp.feature_names)
    if interaction_sample_n and interaction_sample_n > 0 and len(X) > interaction_sample_n:
        Xs = X.sample(interaction_sample_n, random_state=123)
    else:
        Xs = X
    inter = None
    if explainer is not None:
        try:
            inter = explainer.shap_interaction_values(Xs)
            if isinstance(inter, list):
                inter = inter[0]
        except Exception as e:
            log_step(f"[WARN] TreeExplainer交互失败，改用XGBoost原生pred_interactions: {e}")

    if inter is None:
        booster = model.get_booster()
        dmat = xgboost.DMatrix(Xs, feature_names=list(Xs.columns))
        raw_inter = booster.predict(dmat, pred_interactions=True, validate_features=False)
        inter = raw_inter[:, :-1, :-1]
    mat = np.abs(inter).mean(axis=0)
    feats = list(Xs.columns)

    rows = []
    p = len(feats)
    for i in range(p):
        for j in range(i + 1, p):
            rows.append((feats[i], feats[j], float(mat[i, j])))
    top_df = pd.DataFrame(rows, columns=["feature_i", "feature_j", "mean_abs_interaction"]).sort_values(
        "mean_abs_interaction", ascending=False
    )
    top_df.to_csv(os.path.join(out_dir, f"interaction_top_pairs_{y_col}.csv"), index=False, encoding="utf-8-sig")

    main_imp = np.abs(exp.values).mean(axis=0)
    top_idx = np.argsort(-main_imp)[: min(20, len(feats))]
    sub = mat[np.ix_(top_idx, top_idx)]
    sub_feats = [feats[i] for i in top_idx]
    plt.figure(figsize=(10, 8))
    im = plt.imshow(sub, cmap="YlOrRd")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(sub_feats)), sub_feats, rotation=80)
    plt.yticks(range(len(sub_feats)), sub_feats)
    plt.title(f"SHAP Interaction Heatmap ({y_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"interaction_heatmap_{y_col}.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log_step(f"[OK] 交互SHAP完成: {y_col} | top输出={INTERACTION_TOP_K}")


def run_temporal_shap(exp, df_used: pd.DataFrame, y_col: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tmp = pd.DataFrame(exp.values, columns=exp.feature_names)
    tmp[YEAR_COL] = df_used[YEAR_COL].values

    long_rows = []
    signed_rows = []
    for yr, g in tmp.groupby(YEAR_COL):
        m = g.drop(columns=[YEAR_COL]).abs().mean(axis=0)
        part = pd.DataFrame({"年份": yr, "feature": m.index, "mean_abs_shap": m.values})
        long_rows.append(part)
        s = g.drop(columns=[YEAR_COL]).mean(axis=0)
        part_s = pd.DataFrame({"年份": yr, "feature": s.index, "mean_shap": s.values})
        signed_rows.append(part_s)
    long_df = pd.concat(long_rows, ignore_index=True).sort_values(["feature", "年份"])
    signed_long_df = pd.concat(signed_rows, ignore_index=True).sort_values(["feature", "年份"])
    wide_df = long_df.pivot_table(index="年份", columns="feature", values="mean_abs_shap", aggfunc="mean")
    signed_wide_df = signed_long_df.pivot_table(index="年份", columns="feature", values="mean_shap", aggfunc="mean")
    long_df.to_csv(os.path.join(out_dir, f"time_meanabs_long_{y_col}.csv"), index=False, encoding="utf-8-sig")
    wide_df.to_csv(os.path.join(out_dir, f"time_meanabs_wide_{y_col}.csv"), encoding="utf-8-sig")
    signed_long_df.to_csv(os.path.join(out_dir, f"time_meansigned_long_{y_col}.csv"), index=False, encoding="utf-8-sig")
    signed_wide_df.to_csv(os.path.join(out_dir, f"time_meansigned_wide_{y_col}.csv"), encoding="utf-8-sig")

    overall = long_df.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False)
    top_feats = overall.head(MONITOR_TOP_K).index.tolist()
    plt.figure(figsize=(10, 6))
    for f in top_feats:
        d = long_df[long_df["feature"] == f].sort_values("年份")
        plt.plot(d["年份"], d["mean_abs_shap"], marker="o", linewidth=1.6, label=f)
    plt.title(f"Yearly SHAP Drift ({y_col})")
    plt.xlabel("年份")
    plt.ylabel("mean(|SHAP|)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"time_drift_lines_{y_col}.png"), dpi=180, bbox_inches="tight")
    plt.close()

    exp_sort = exp[np.argsort(df_used[YEAR_COL].values)]
    mon_ok = False
    for f in top_feats[: min(3, len(top_feats))]:
        try:
            fi = exp.feature_names.index(f)
            plt.figure(figsize=(9, 5))
            try:
                shap.plots.monitoring(fi, exp_sort, show=False)
            except Exception:
                shap.plots.monitoring(fi, exp_sort.values, exp_sort.data, show=False)
            plt.title(f"Monitoring: {f} ({y_col})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"monitoring_{y_col}_{f}.png"), dpi=180, bbox_inches="tight")
            plt.close()
            mon_ok = True
        except Exception as e:
            plt.close()
            log_step(f"[WARN] monitoring失败({f}, {y_col}): {e}")
    if mon_ok:
        log_step(f"[OK] monitoring图已输出: {y_col}")
    log_step(f"[OK] 时间SHAP完成: {y_col}")


def main():
    parser = argparse.ArgumentParser(description="SHAP扩展分析：局部/分组差异/交互/时间")
    parser.add_argument("--path", default=DEFAULT_PATH, help="输入CSV路径")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--debug-n", type=int, default=0, help="小样本调试行数，<=0表示全量")
    parser.add_argument("--debug-seed", type=int, default=123, help="小样本抽样随机种子")
    parser.add_argument("--interaction-sample-n", type=int, default=INTERACTION_SAMPLE_N,
                        help="交互SHAP抽样行数，<=0表示全量")
    parser.add_argument("--only-y", default="", help="仅运行指定目标列名，空表示全部")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_chinese_font()

    log_step(f"[INFO] 读取数据: {args.path}")
    log_step(f"[INFO] 输出目录: {args.out_dir}")
    df, x_lag_cols, y_cols = prepare_data(args.path)
    if args.debug_n and args.debug_n > 0 and len(df) > args.debug_n:
        df = df.sample(args.debug_n, random_state=args.debug_seed).sort_values([ID_COL, YEAR_COL]).reset_index(
            drop=True)
        log_step(f"[INFO] 小样本调试模式开启: n={len(df)} (debug_n={args.debug_n}, seed={args.debug_seed})")

    if args.only_y:
        y_cols = [y for y in y_cols if y == args.only_y]
        if not y_cols:
            raise ValueError(f"--only-y 指定列不存在或不可用: {args.only_y}")

    for y_col in y_cols:
        log_step(f"[INFO] 开始目标变量: {y_col}")
        y_dir = os.path.join(args.out_dir, f"y_{y_col}")
        os.makedirs(y_dir, exist_ok=True)

        model, explainer, exp, pred, df_used = fit_xgb(df, x_lag_cols, y_col)
        pd.DataFrame({
            "feature": exp.feature_names,
            "mean_abs_shap": np.abs(exp.values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            os.path.join(y_dir, f"global_meanabs_{y_col}.csv"), index=False, encoding="utf-8-sig"
        )
        signed_df = pd.DataFrame({
            "feature": exp.feature_names,
            "mean_shap": exp.values.mean(axis=0),
            "mean_abs_shap": np.abs(exp.values).mean(axis=0),
            "pos_rate": (exp.values > 0).mean(axis=0),
            "neg_rate": (exp.values < 0).mean(axis=0),
        }).sort_values("mean_shap", ascending=False)
        signed_df.to_csv(
            os.path.join(y_dir, f"global_meansigned_{y_col}.csv"), index=False, encoding="utf-8-sig"
        )
        log_step(f"[OK] 已输出方向性SHAP: {os.path.join(y_dir, f'global_meansigned_{y_col}.csv')}")

        run_local_waterfall(exp, df_used, y_col, pred, os.path.join(y_dir, "01_local_waterfall"))
        run_cohorts_and_group_diff(exp, df_used, y_col, os.path.join(y_dir, "02_cohorts_groupdiff"))
        run_interaction_shap(
            model,
            explainer,
            exp,
            df_used,
            y_col,
            os.path.join(y_dir, "03_interaction"),
            interaction_sample_n=args.interaction_sample_n,
        )
        run_temporal_shap(exp, df_used, y_col, os.path.join(y_dir, "04_temporal_monitoring"))

        log_step(f"[INFO] 目标变量完成: {y_col}")

    log_step("[INFO] 全部扩展SHAP分析完成")


if __name__ == "__main__":
    main()
