# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from econml.dml import LinearDML, SparseLinearDML, CausalForestDML

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

EN_DICT = {
    "裁判文书说理平均字数": "Avg Words in Judgment",
    "一审上诉发回重审率": "Remand Rate (1st Inst Appeal)",
    "案件数占人口比例": "Cases per Capita",
    "一审上诉改判率": "Commutation Rate (1st Inst)",
    "案件平均审理天数": "Avg Trial Days",
    "生效案件改判率": "Commutation Rate (Effective Cases)",
    "案件发回重审率": "Remand Rate",
    "人均律师数量": "Lawyers per Capita",
    "案件总数": "Total Cases",
    "一审陪审率": "Jury Trial Rate (1st Inst)",
    "二审开庭率": "Hearing Rate (2nd Inst)",
    "独任审理率": "Sole-judge Trial Rate",
    "上诉率": "Appeal Rate",
    "再审率": "Retrial Rate",

    "年份": "Year", "省份": "Province", "平均因果效应": "Marginal Effect (CATE)",
    "地区生产总值增长率-全市(%)": "GDP Growth Rate (%)",
    "地方一般公共预算收入（万元）全市": "Local Public Budget Revenue",
    "第一产业占地区生产总值的比重-全市": "Primary Industry Share (%)",
    "第二产业占地区生产总值的比重-全市": "Secondary Industry Share (%)",
    "规模以上工业企业主要财务指标_利润总额（万元）全市": "Industrial Enterprises Profit",
    "规模以上工业企业数（个）全市": "Number of Industrial Enterprises",
    "工业二氧化硫排放量（吨）全市": "Industrial SO2 Emissions",
    "污水处理厂集中处理率（%）全市": "Sewage Treatment Rate (%)",
    "电信业务收入（万元）全市": "Telecom Revenue",
    "邮政业务收入（万元）全市": "Postal Revenue",
    "限额以上批发零售业商品销售总额（万元）全市": "Wholesale and Retail Sales",
    "限额以上批发零售商贸企业数_法人数（个）全市": "Num of Wholesale/Retail Enterprises",
    "城镇职工基本医疗保险参保人数（人）全市": "Medical Insurance Participants",
    "普通中学（所）全市": "Number of Middle Schools",
    "普通小学专任教师数（人）全市": "Primary School Teachers",
    "普通高等学校专任教师数（人）全市": "Higher Education Teachers",
    "中等职业教育学校在校学生数（人）全市": "Secondary Vocational Students",

    "外商直接投资（百万美元）": "FDI (in millions of USD)",
    "外商直接投资额（百万元人民币）": "FDI (in millions of RMB)"
}


def t(word):
    return EN_DICT.get(word, "Unknown Variable")


def get_en(word):
    return EN_DICT.get(word, word).replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("%",
                                                                                                                "pct")


PATH = r"Dataset.csv"
LOG_DIR = r"dml_results"

PLOT_DIRS = {
    "SHAP": os.path.join(LOG_DIR, "1_plots_SHAP_Controls"),
    "TIME": os.path.join(LOG_DIR, "2_plots_Time_Hetero"),
    "SPACE": os.path.join(LOG_DIR, "3_plots_Province_Hetero"),
    "MAP": os.path.join(LOG_DIR, "4_plots_China_Map")
}
for d in PLOT_DIRS.values():
    os.makedirs(d, exist_ok=True)

Y_COLS = [
    "外商直接投资（百万美元）",
    "外商直接投资额（百万元人民币）",
]
TREATMENTS = [
    "案件总数",
    "案件数占人口比例",
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
GROUP_COL, PROVINCE_COL, TIME_COL = "城市", "省份", "年份"

ALL_CONTROLS = [
    "地区生产总值增长率-全市(%)",
    "地方一般公共预算收入（万元）全市",
    "第一产业占地区生产总值的比重-全市",
    "第二产业占地区生产总值的比重-全市",
    "规模以上工业企业主要财务指标_利润总额（万元）全市",
    "规模以上工业企业数（个）全市",
    "工业二氧化硫排放量（吨）全市",
    "污水处理厂集中处理率（%）全市",
    "电信业务收入（万元）全市",
    "邮政业务收入（万元）全市",
    "限额以上批发零售业商品销售总额（万元）全市",
    "限额以上批发零售商贸企业数_法人数（个）全市",
    "城镇职工基本医疗保险参保人数（人）全市",
    "普通中学（所）全市",
    "普通小学专任教师数（人）全市",
    "普通高等学校专任教师数（人）全市",
    "中等职业教育学校在校学生数（人）全市"
]


def plot_time_heterogeneity(df_result, y_name, t_name, model_name):
    time_cate = df_result.groupby(TIME_COL)['CATE'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=time_cate, x=TIME_COL, y='CATE', marker='o', linewidth=2.5, color='#1f77b4')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)

    plt.title(f"Temporal Heterogeneity [{model_name}]\n{t(t_name)} -> {t(y_name)}", fontsize=13, pad=15)
    plt.ylabel(t("平均因果效应"), fontsize=12)
    plt.xlabel(t("年份"), fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRS["TIME"], f"Time_{model_name}_{get_en(t_name)}_{get_en(y_name)}.png"), dpi=300)
    plt.close()


def plot_space_heterogeneity(df_result, y_name, t_name, model_name):
    prov_cate = df_result.groupby(PROVINCE_COL)['CATE'].mean().reset_index()
    prov_cate = prov_cate.sort_values(by='CATE', ascending=False)

    PROV_EN_DICT = {
        "北京市": "Beijing", "天津市": "Tianjin", "河北省": "Hebei", "山西省": "Shanxi",
        "内蒙古自治区": "Inner Mongolia",
        "辽宁省": "Liaoning", "吉林省": "Jilin", "黑龙江省": "Heilongjiang", "上海市": "Shanghai", "江苏省": "Jiangsu",
        "浙江省": "Zhejiang", "安徽省": "Anhui", "福建省": "Fujian", "江西省": "Jiangxi", "山东省": "Shandong",
        "河南省": "Henan", "湖北省": "Hubei", "湖南省": "Hunan", "广东省": "Guangdong", "广西壮族自治区": "Guangxi",
        "海南省": "Hainan", "重庆市": "Chongqing", "四川省": "Sichuan", "贵州省": "Guizhou", "云南省": "Yunnan",
        "西藏自治区": "Tibet", "陕西省": "Shaanxi", "甘肃省": "Gansu", "青海省": "Qinghai", "宁夏回族自治区": "Ningxia",
        "新疆维吾尔自治区": "Xinjiang", "台湾省": "Taiwan", "香港特别行政区": "Hong Kong", "澳门特别行政区": "Macau",
        "北京": "Beijing", "天津": "Tianjin", "河北": "Hebei", "山西": "Shanxi", "内蒙古": "Inner Mongolia",
        "辽宁": "Liaoning", "吉林": "Jilin", "黑龙江": "Heilongjiang", "上海": "Shanghai", "江苏": "Jiangsu",
        "浙江": "Zhejiang", "安徽": "Anhui", "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong",
        "河南": "Henan", "湖北": "Hubei", "湖南": "Hunan", "广东": "Guangdong", "广西": "Guangxi",
        "海南": "Hainan", "重庆": "Chongqing", "四川": "Sichuan", "贵州": "Guizhou", "云南": "Yunnan",
        "西藏": "Tibet", "陕西": "Shaanxi", "甘肃": "Gansu", "青海": "Qinghai", "宁夏": "Ningxia", "新疆": "Xinjiang"
    }

    prov_cate['Province_EN'] = prov_cate[PROVINCE_COL].map(lambda x: PROV_EN_DICT.get(str(x).strip(), str(x)))

    csv_path = os.path.join(PLOT_DIRS["SPACE"], f"Province_Effects_{model_name}_{get_en(t_name)}_{get_en(y_name)}.csv")
    prov_cate.rename(columns={'CATE': 'Marginal_Effect_CATE'}).to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, max(8, len(prov_cate) * 0.3)))
    sns.barplot(data=prov_cate, x='CATE', y='Province_EN', palette="RdBu_r")

    plt.title(f"Provincial Heterogeneity [{model_name}]\n{t(t_name)} -> {t(y_name)}", fontsize=13, pad=15)
    plt.xlabel(t("平均因果效应"), fontsize=12)
    plt.ylabel(t("省份"), fontsize=12)
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRS["SPACE"], f"Province_{model_name}_{get_en(t_name)}_{get_en(y_name)}.png"),
                dpi=300)
    plt.close()


def plot_shap_analysis(est, X_data, X_names, y_name, t_name, model_name):
    try:
        shap_values_dict = est.shap_values(X_data)
        y_key = list(shap_values_dict.keys())[0]
        t_key = list(shap_values_dict[y_key].keys())[0]
        shap_vals = shap_values_dict[y_key][t_key]

        en_names = [t(name) for name in X_names]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_data, feature_names=en_names, show=False)
        plt.title(
            f"SHAP Summary: Control Variables Impact [{model_name}]\nTreatment: {t(t_name)} -> Outcome: {t(y_name)}",
            fontsize=13, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIRS["SHAP"], f"SHAP_{model_name}_{get_en(t_name)}_{get_en(y_name)}.png"),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
    except Exception as e:
        pass


def plot_china_map(df_result, y_name, t_name, model_name):
    try:
        prov_cate = df_result.groupby(PROVINCE_COL)['CATE'].mean().reset_index()

        DATAV_NAMES = {
            "北京": "北京市", "天津": "天津市", "河北": "河北省", "山西": "山西省", "内蒙古": "内蒙古自治区",
            "辽宁": "辽宁省", "吉林": "吉林省", "黑龙江": "黑龙江省", "上海": "上海市", "江苏": "江苏省",
            "浙江": "浙江省", "安徽": "安徽省", "福建": "福建省", "江西": "江西省", "山东": "山东省",
            "河南": "河南省", "湖北": "湖北省", "湖南": "湖南省", "广东": "广东省", "广西": "广西壮族自治区",
            "海南": "海南省", "重庆": "重庆市", "四川": "四川省", "贵州": "贵州省", "云南": "云南省",
            "西藏": "西藏自治区", "陕西": "陕西省", "甘肃": "甘肃省", "青海": "青海省", "宁夏": "宁夏回族自治区",
            "新疆": "新疆维吾尔自治区", "台湾": "台湾省", "香港": "香港特别行政区", "澳门": "澳门特别行政区"
        }

        def to_datav_name(n):
            n = str(n).strip()
            for short, full in DATAV_NAMES.items():
                if short in n:
                    return full
            return n

        prov_cate['Map_Name'] = prov_cate[PROVINCE_COL].apply(to_datav_name)

        url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
        china_map = gpd.read_file(url)

        merged = china_map.merge(prov_cate, left_on='name', right_on='Map_Name', how='left')

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        v_min, v_max = float(prov_cate['CATE'].min()), float(prov_cate['CATE'].max())
        limit = max(abs(v_min), abs(v_max))
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        china_map.plot(ax=ax, color='#eeeeee', edgecolor='black', linewidth=0.5)
        merged.dropna(subset=['CATE']).plot(
            column='CATE', ax=ax, cmap='RdBu_r', norm=norm,
            edgecolor='black', linewidth=0.5,
            legend=True,
            legend_kwds={
                'label': "Marginal Effect (CATE)",
                'orientation': "horizontal",
                'shrink': 0.6,
                'pad': 0.05
            }
        )

        ax.axis('off')
        plt.title(f"Spatial Distribution of CATE [{model_name}]\n{t(t_name)} -> {t(y_name)}", fontsize=14, pad=15)
        png_path = os.path.join(PLOT_DIRS["MAP"], f"Map_{model_name}_{get_en(t_name)}_{get_en(y_name)}.png")
        pdf_path = os.path.join(PLOT_DIRS["MAP"], f"Map_{model_name}_{get_en(t_name)}_{get_en(y_name)}.pdf")

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')  # PDF is optimal for vector graphics
        plt.close()

    except Exception as e:
        print(f"      [Map Skipped] Static map drawing failed for {model_name}: {e}")


def get_base_models(model_name):
    if model_name == "Ridge":
        return Ridge(alpha=1.0), Ridge(alpha=1.0)
    elif model_name == "RF":
        return RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42), RandomForestRegressor(
            n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
    elif model_name == "GBDT":
        return GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "CAT":
        return CatBoostRegressor(iterations=100, depth=5, random_state=42, verbose=0), CatBoostRegressor(iterations=100,
                                                                                                         depth=5,
                                                                                                         random_state=42,
                                                                                                         verbose=0)


def main():
    print("Loading and executing strict data preprocessing...")
    df = pd.read_csv(PATH)

    actual_controls = [c for c in ALL_CONTROLS if c in df.columns]
    df = df.dropna(subset=Y_COLS + TREATMENTS + actual_controls + [GROUP_COL, PROVINCE_COL, TIME_COL]).reset_index(
        drop=True)

    scaler_w = StandardScaler()
    Control_data = scaler_w.fit_transform(df[actual_controls].values)
    groups = df[GROUP_COL].values

    ate_results = []
    base_model_names = ["Ridge", "RF", "GBDT", "CAT"]
    dml_estimators = ["Linear", "SparseLinear", "Forest"]

    for y in Y_COLS:
        Y_data = np.log1p(df[y].values)

        for t_var in TREATMENTS:
            T_raw = df[t_var].values.reshape(-1, 1)
            scaler_t = StandardScaler()
            T_data = scaler_t.fit_transform(T_raw).flatten()

            print(f"\n[{y} (log)] <--- Influenced by ---> [{t_var} (std)]")

            ensemble_cates_list = []
            ensemble_shap_list = []

            for base_name in base_model_names:
                for est_type in dml_estimators:
                    model_comb = f"{base_name}_{est_type}"
                    try:
                        model_y, model_t = get_base_models(base_name)
                        cv = GroupKFold(n_splits=5)

                        if est_type == "Linear":
                            est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, cv=cv,
                                            random_state=42)
                        elif est_type == "SparseLinear":
                            est = SparseLinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, cv=cv,
                                                  random_state=42)
                        elif est_type == "Forest":
                            est = CausalForestDML(
                                model_y=model_y,
                                model_t=model_t,
                                discrete_treatment=False,
                                cv=cv,
                                random_state=42,
                                n_jobs=-1,
                                n_estimators=2000,
                                min_samples_leaf=30,
                                max_depth=5
                            )

                        est.fit(Y=Y_data, T=T_data, X=Control_data, W=None, groups=groups)

                        inf = est.const_marginal_ate_inference(X=Control_data)
                        ate = float(np.squeeze(inf.mean_point))
                        se = float(np.squeeze(inf.stderr_mean))

                        cates = np.squeeze(est.const_marginal_effect(X=Control_data))
                        ensemble_cates_list.append(cates)

                        try:
                            shap_values_dict = est.shap_values(Control_data)
                            y_key = list(shap_values_dict.keys())[0]
                            t_key = list(shap_values_dict[y_key].keys())[0]
                            shap_vals = shap_values_dict[y_key][t_key]

                            if hasattr(shap_vals, 'values'):
                                shap_vals = shap_vals.values

                            ensemble_shap_list.append(shap_vals)
                        except Exception as e:
                            pass

                        df_hetero = df[[PROVINCE_COL, TIME_COL]].copy()
                        df_hetero['CATE'] = cates

                        plot_time_heterogeneity(df_hetero, y, t_var, model_comb)
                        plot_space_heterogeneity(df_hetero, y, t_var, model_comb)
                        plot_shap_analysis(est, Control_data, actual_controls, y, t_var, model_comb)
                        plot_china_map(df_hetero, y, t_var, model_comb)

                        z_stat = ate / se if se > 0 else 0
                        p_value = float(2 * (1 - st.norm.cdf(abs(z_stat))))
                        ate_results.append({
                            "Outcome_Y": y, "Treatment_T": t_var,
                            "Base_Model": base_name, "DML_Framework": est_type,
                            "ATE": ate, "SE": se, "p_value": p_value
                        })
                        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                        print(f"  [{model_comb:18s}] ATE: {ate:8.4f} | p-val: {p_value:.4f} {sig}")

                    except Exception as e:
                        print(f"  [Error] {model_comb} execution failed: {e}")

            if ensemble_cates_list:
                mean_cates = np.mean(ensemble_cates_list, axis=0)
                df_hetero_avg = df[[PROVINCE_COL, TIME_COL]].copy()
                df_hetero_avg['CATE'] = mean_cates

                print(f"  [Ensemble] Generating cross-model average plots for {t_var} -> {y}...")
                plot_china_map(df_hetero_avg, y, t_var, "Ensemble_Average")
                plot_space_heterogeneity(df_hetero_avg, y, t_var, "Ensemble_Average")
                plot_time_heterogeneity(df_hetero_avg, y, t_var, "Ensemble_Average")

            if ensemble_shap_list:
                mean_shap_vals = np.mean(ensemble_shap_list, axis=0)
                en_names = [t(name) for name in actual_controls]

                print(f"  [Ensemble] Generating SHAP summary plot and feature importance CSV for {t_var} -> {y}...")

                plt.figure(figsize=(12, 8))
                shap.summary_plot(mean_shap_vals, Control_data, feature_names=en_names, show=False)
                plt.title(
                    f"SHAP Summary: Control Variables Impact [Ensemble_Average]\nTreatment: {t(t_var)} -> Outcome: {t(y)}",
                    fontsize=13, pad=20)
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIRS["SHAP"], f"SHAP_Ensemble_Average_{get_en(t_var)}_{get_en(y)}.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()

                mean_abs_shap = np.abs(mean_shap_vals).mean(axis=0)
                shap_df = pd.DataFrame({
                    'Feature': en_names,
                    'Mean_Abs_SHAP': mean_abs_shap
                }).sort_values(by='Mean_Abs_SHAP', ascending=False)

                csv_path = os.path.join(PLOT_DIRS["SHAP"],
                                        f"SHAP_Importance_Ensemble_Average_{get_en(t_var)}_{get_en(y)}.csv")
                shap_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    res_df = pd.DataFrame(ate_results)
    save_path = os.path.join(LOG_DIR, "Main_ATE_Results.csv")
    res_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nAll calculations completed!\nMain ATE results table saved to: {save_path}")


if __name__ == "__main__":
    main()
