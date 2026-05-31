"""
Smart-E Robo — 三階段 ETF 篩選與 MVO 優化 Pipeline
=======================================================
Stage 0  人工預篩（市值排序）
  - 318 檔 ETF，各類型按市值由大到小排序
  - 按各類型佔全體比例分配 100 個名額（每類至少 10 檔）
  - 股票型: 43 檔 / 債券型: 34 檔 / 其他型: 23 檔

Stage 1  量化評分
  - ADV（流動性）、Active Risk（主動風險）、Cost（總費用率）
  - 百分比排名法，三項等權加總

Stage 2  優選池 → MVO
  - 取綜合分數前 10%（共 10 檔），債券+其他合計至少 2 檔
  - MVO 最大化夏普比率，產出最佳權重

回測
  - Smart-E 10 檔加權組合 vs. 100 檔等權組合

Benchmark:
  股票型 → ^TWII
  債券型 → 00679B.TW
  其他型 → VT
"""

import json, math, os, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 0. 原始資料（來自 ETF分類表.xlsx）
# ══════════════════════════════════════════════════════════════

RAW_STOCK = [
    ("0050",18898.22,0.075),("0051",34.47,0.435),("0052",1480.84,0.185),
    ("0053",11.79,0.44),("0055",27.87,0.435),("0056",6626.21,0.335),
    ("0057",4.43,0.185),("0061",25.97,0.4),("006201",9.83,0.435),
    ("006203",18.72,0.335),("006204",3.28,0.355),("006205",42.23,1.09),
    ("006206",10.89,1.09),("006207",7.58,0.85),("006208",4273.87,0.13),
    ("00636",30.87,1.05),("00639",20.44,1.09),("00643",16.74,1.08),
    ("00645",28.31,0.7),("00646",409.43,0.51),("00652",22.38,1.25),
    ("00657",5.06,0.65),("00660",3.61,0.74),("00661",20.08,0.7),
    ("00662",896.77,0.46),("00668",6.71,0.63),("00678",2.77,1.01),
    ("00690",63.12,0.355),("00692",540.32,0.185),("00700",6.16,0.68),
    ("00701",59.97,0.335),("00702",1.63,0.5),("00703",1.97,1.1),
    ("00709",6.12,0.79),("00713",1061.8,0.385),("00714",4.78,1.03),
    ("00717",19.78,0.8),("00728",24.31,0.435),("00730",20.98,0.385),
    ("00731",19.21,0.485),("00733",56.0,0.335),("00735",73.14,0.48),
    ("00736",1.93,0.66),("00737",8.69,1.1),("00739",7.6,0.8),
    ("00752",71.07,1.17),("00757",376.83,1.03),("00762",54.22,1.16),
    ("00770",54.39,0.35),("00771",2.24,0.75),("00783",3.23,1.09),
    ("00830",544.8,0.41),("00850",362.03,0.335),("00851",4.14,1.0),
    ("00858",43.44,0.51),("00861",60.71,1.13),("00875",6.28,1.1),
    ("00876",66.58,1.13),("00877",57.99,1.0),("00878",5636.76,0.285),
    ("00881",1258.4,0.435),("00882",322.03,0.61),("00885",125.45,1.22),
    ("00886",1.96,0.56),("00887",69.07,0.87),("00888",127.95,0.28),
    ("00891",650.61,0.435),("00892",119.13,0.435),("00893",173.09,1.1),
    ("00894",76.93,0.435),("00895",78.98,1.1),("00896",75.9,0.435),
    ("00897",14.98,1.05),("00898",13.06,1.05),("00899",4.62,1.1),
    ("00900",324.69,0.335),("00901",29.54,0.435),("00902",40.04,1.1),
    ("00903",11.64,1.0),("00904",71.0,0.435),("00905",88.76,0.335),
    ("00907",27.73,0.485),("00909",69.09,0.95),("00910",76.39,1.13),
    ("00911",10.54,1.05),("00912",18.82,0.435),("00913",9.7,0.49),
    ("00915",171.92,0.285),("00916",17.54,0.78),("00917",14.08,1.1),
    ("00918",865.14,0.385),("00919",4918.88,0.335),("00920",5.47,1.1),
    ("00921",13.47,0.49),("00922",770.88,0.235),("00923",345.31,0.355),
    ("00924",146.41,0.51),("00926",22.57,0.83),("00927",297.58,0.435),
    ("00928",11.85,0.435),("00929",1374.38,0.33),("00930",34.07,0.435),
    ("00932",26.38,0.49),("00934",108.96,0.335),("00935",351.87,0.435),
    ("00936",50.68,0.435),("00938",17.45,0.385),("00939",248.22,0.335),
    ("00940",694.88,0.33),("00941",115.84,1.1),("00943",9.16,0.49),
    ("00944",11.53,0.435),("00946",70.66,0.33),("00947",75.98,0.435),
    ("00949",40.18,0.75),("00951",37.13,0.65),("00952",42.42,0.435),
    ("00954",18.01,0.75),("00955",91.55,0.75),("00956",4.29,0.65),
    ("00960",12.01,1.1),("00961",20.21,0.485),("00962",6.99,0.435),
    ("00963",11.3,0.9),("00964",9.51,0.9),("00965",134.1,1.04),
    ("00971",11.79,0.75),("00972",3.91,0.75),("009800",37.5,0.44),
    ("009801",14.22,0.42),("009802",101.98,0.33),("009803",28.05,0.335),
    ("009804",26.85,0.185),("009805",117.23,0.83),("009806",2.0,0.45),
    ("009807",2.54,0.45),("009808",12.1,0.19),("009809",5.4,0.33),
    ("009810",4.09,1.0),("009811",37.47,0.47),("009812",13.73,0.37),
    ("009813",51.14,0.47),("009814",19.35,0.36),("009815",66.13,0.42),
    ("009816",1278.34,0.097),("009818",11.86,0.47),("009819",66.29,0.8),
    ("009820",180.2,0.33),
]

RAW_BOND = [
    ("00679B",1913.97,0.14),("00680L",127.09,0.96),("00681R",1.61,0.91),
    ("00687B",1542.54,0.14),("00687C",1597.08,0.15),("00688L",52.85,0.86),
    ("00689R",1.53,0.81),("00694B",17.22,0.22),("00695B",13.43,0.37),
    ("00696B",191.25,0.16),("00697B",23.78,0.3),("00710B",34.58,0.4),
    ("00711B",138.34,0.45),("00719B",165.73,0.13),("00720B",1366.44,0.35),
    ("00722B",524.06,0.26),("00723B",464.71,0.26),("00724B",1032.21,0.26),
    ("00725B",1312.75,0.33),("00726B",388.83,0.32),("00727B",22.24,0.56),
    ("00734B",4.19,0.51),("00740B",821.95,0.28),("00741B",5.42,0.56),
    ("00746B",877.16,0.23),("00749B",344.3,0.28),("00750B",48.46,0.3),
    ("00751B",1662.8,0.23),("00754B",465.97,0.26),("00755B",91.33,0.4),
    ("00756B",462.3,0.31),("00758B",2.82,0.51),("00759B",25.58,0.51),
    ("00760B",229.91,0.31),("00761B",934.87,0.23),("00764B",431.71,0.16),
    ("00768B",366.82,0.15),("00772B",1146.13,0.28),("00773B",908.41,0.26),
    ("00775B",73.76,0.54),("00777B",610.41,0.23),("00778B",513.79,0.25),
    ("00779B",363.43,0.13),("00780B",98.8,0.4),("00781B",2.37,0.45),
    ("00782B",5.85,0.45),("00785B",373.79,0.26),("00786B",7.46,0.4),
    ("00787B",4.15,0.4),("00788B",8.11,0.4),("00789B",218.49,0.26),
    ("00791B",156.92,0.33),("00792B",720.13,0.25),("00793B",1.08,0.45),
    ("00795B",463.2,0.13),("00799B",7.45,0.41),("00834B",7.83,0.35),
    ("00836B",166.59,0.26),("00840B",17.64,0.56),("00841B",20.15,0.56),
    ("00842B",4.91,0.35),("00844B",71.97,0.3),("00845B",77.76,0.4),
    ("00846B",26.29,0.57),("00847B",2.21,0.4),("00848B",46.15,0.44),
    ("00849B",179.41,0.37),("00853B",50.75,0.34),("00856B",2.56,0.2),
    ("00857B",134.37,0.16),("00859B",8.98,0.14),("00860B",1.2,0.26),
    ("00862B",264.79,0.36),("00863B",184.12,0.34),("00864B",44.19,0.16),
    ("00865B",113.34,0.14),("00867B",160.16,0.26),("00870B",157.78,0.3),
    ("00883B",1.32,0.54),("00884B",22.16,0.46),("00890B",6.17,0.43),
    ("00931B",291.49,0.12),("00933B",1061.83,0.25),("00937B",2595.77,0.25),
    ("00942B",292.24,0.23),("00945B",123.04,0.36),("00948B",390.05,0.26),
    ("00950B",563.99,0.25),("00953B",421.34,0.36),("00957B",30.61,0.4),
    ("00958B",13.87,0.47),("00959B",51.61,0.38),("00966B",17.91,0.39),
    ("00967B",17.83,0.12),("00968B",60.73,0.23),("00969B",15.98,0.12),
    ("00970B",37.59,0.3),("00980B",5.27,0.4),("00981B",129.55,0.36),
    ("00981D",60.36,0.0),("00982B",2.94,0.4),("00983B",4.92,0.22),
    ("00984B",27.6,0.4),("00985B",52.06,0.31),("00986B",2.74,0.47),
    ("00986D",29.1,0.0),("00987B",13.24,0.33),("00988B",16.15,0.51),
    ("00989B",6.57,0.47),
]

RAW_OTHER = [
    ("00400A",223.65,0.94),("00401A",33.84,0.645),("00631L",1581.48,1.04),
    ("00632R",243.7,1.04),("00633L",73.49,1.22),("00634R",1.46,1.22),
    ("00637L",176.81,1.22),("00638R",1.81,1.2),("00640L",9.59,1.2),
    ("00641R",2.89,1.17),("00647L",8.73,1.19),("00648R",5.82,1.17),
    ("00650L",20.8,1.12),("00651R",1.75,1.12),("00653L",8.55,1.22),
    ("00654R",1.33,1.22),("00655L",26.41,1.16),("00656R",1.19,1.13),
    ("00663L",113.49,0.79),("00664R",13.36,0.79),("00665L",45.45,1.02),
    ("00666R",1.34,1.02),("00669R",11.24,1.03),("00670L",159.22,1.09),
    ("00671R",13.23,1.07),("00675L",337.71,0.69),("00676R",10.2,0.69),
    ("00685L",116.03,0.0),("00686R",1.89,0.34),("00712",431.38,0.33),
    ("00753L",71.88,1.19),("00852L",2.94,1.03),("00908",5.34,0.83),
    ("00980A",185.71,0.785),("00980D",21.26,0.0),("00980T",8.92,0.98),
    ("009817",25.01,0.32),("00981A",2852.74,0.1),("00981T",21.0,0.0),
    ("00982A",560.85,0.835),("00982D",8.2,0.65),("00982T",2.8,0.98),
    ("00983A",25.01,0.15),("00983D",10.54,0.75),("00984A",71.7,0.74),
    ("00984D",31.0,0.88),("00985A",112.31,0.485),("00985D",7.33,0.67),
    ("00986A",6.35,1.25),("00987A",36.92,0.785),("00988A",419.79,0.1),
    ("00989A",14.96,0.9),("00990A",344.26,1.05),("00991A",513.58,0.835),
    ("00992A",547.86,0.035),("00993A",120.93,0.835),("00994A",58.04,0.735),
    ("00995A",59.03,0.785),("00996A",50.53,0.84),("00997A",127.58,0.1),
    ("00998A",27.82,0.0),
]

BENCHMARKS = {"stock": "^TWII", "bond": "00679B.TW", "other": "VT"}


# ══════════════════════════════════════════════════════════════
# STAGE 0：市值預篩 → 100 檔
# ══════════════════════════════════════════════════════════════

def stage0_market_cap_filter():
    """各類型按市值排序，按比例分配 100 名額（每類≥10），產出 100 檔候選清單"""
    cats = {
        "stock": sorted(RAW_STOCK, key=lambda x: x[1], reverse=True),
        "bond":  sorted(RAW_BOND,  key=lambda x: x[1], reverse=True),
        "other": sorted(RAW_OTHER, key=lambda x: x[1], reverse=True),
    }
    total_etfs = sum(len(v) for v in cats.values())  # 318
    total_slots = 100
    min_slots   = 10

    # 先給每類 min_slots，剩餘 70 個按比例分
    remaining = total_slots - min_slots * len(cats)
    slots = {}
    for cat, lst in cats.items():
        prop = len(lst) / total_etfs
        slots[cat] = min_slots + prop * remaining

    # 四捨五入並補齊至 100
    floored  = {k: math.floor(v) for k, v in slots.items()}
    shortage = total_slots - sum(floored.values())
    remainders = {k: slots[k] - floored[k] for k in slots}
    for cat in sorted(remainders, key=remainders.get, reverse=True)[:shortage]:
        floored[cat] += 1

    # 取每類前 N 檔（按市值）
    selected = []
    for cat, lst in cats.items():
        n = floored[cat]
        for ticker, mktcap, cost in lst[:n]:
            selected.append({
                "ticker":   ticker,
                "category": cat,
                "mktcap":   mktcap,
                "cost":     cost,
                "mktcap_rank": lst.index((ticker, mktcap, cost)) + 1,
            })

    df = pd.DataFrame(selected)
    print(f"Stage 0 完成：{floored}，共 {len(df)} 檔")
    return df, floored


# ══════════════════════════════════════════════════════════════
# STAGE 1：量化評分
# ══════════════════════════════════════════════════════════════

def fetch_prices_and_volume(tickers_tw: list, start: str, end: str):
    """批次下載收盤價與成交量"""
    raw = yf.download(tickers_tw, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
        close.columns  = [c.replace(".TW", "") for c in close.columns]
        volume.columns = [c.replace(".TW", "") for c in volume.columns]
    else:
        # 單檔
        ticker = tickers_tw[0].replace(".TW", "")
        close  = raw[["Close"]].rename(columns={"Close": ticker})
        volume = raw[["Volume"]].rename(columns={"Volume": ticker})
    return close, volume


def score_pct_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """百分比排名（0–100），越好分越高"""
    rank = series.rank(pct=True)
    if not higher_is_better:
        rank = 1 - rank
    return (rank * 100).round(2)


def stage1_scoring(df100: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """對 100 檔 ETF 計算三指標評分，回傳含 total_score 的 DataFrame"""
    results = []

    for cat in ["stock", "bond", "other"]:
        sub = df100[df100["category"] == cat].copy()
        tickers = sub["ticker"].tolist()
        tw_tickers = [f"{t}.TW" for t in tickers]

        print(f"  [{cat}] 下載 {len(tickers)} 檔價格資料...")
        close, volume = fetch_prices_and_volume(tw_tickers, start, end)

        # benchmark
        bm_sym = BENCHMARKS[cat]
        bm_raw = yf.download(bm_sym, start=start, end=end,
                             auto_adjust=True, progress=False)["Close"]
        if isinstance(bm_raw, pd.DataFrame):
            bm_raw = bm_raw.iloc[:, 0]
        bm_ret = bm_raw.pct_change().dropna()

        for _, row in sub.iterrows():
            t = row["ticker"]
            if t not in close.columns:
                continue
            px = close[t].dropna()
            if len(px) < 120:   # 資料不足略過
                continue

            # ADV：近一季（63 交易日）平均日成交量（張）
            vol_ser = volume[t] if t in volume.columns else pd.Series(dtype=float)
            adv = vol_ser.iloc[-63:].mean() if len(vol_ser) >= 63 else vol_ser.mean()

            # Active Risk：年化主動風險（%）
            etf_ret = px.pct_change().dropna()
            aligned  = etf_ret.reindex(bm_ret.index).dropna()
            bm_align = bm_ret.reindex(aligned.index)
            excess   = aligned - bm_align
            active_risk = excess.std() * np.sqrt(252) * 100

            results.append({
                "ticker":      t,
                "category":    cat,
                "mktcap":      row["mktcap"],
                "mktcap_rank": row["mktcap_rank"],
                "cost":        row["cost"],
                "ADV":         adv,
                "active_risk": active_risk,
            })

    scored = pd.DataFrame(results).dropna(subset=["ADV", "active_risk"])

    # 全局百分比排名
    scored["score_adv"]  = score_pct_rank(scored["ADV"],         higher_is_better=True)
    scored["score_risk"] = score_pct_rank(scored["active_risk"], higher_is_better=False)
    scored["score_cost"] = score_pct_rank(scored["cost"],        higher_is_better=False)
    scored["total_score"] = (scored["score_adv"] +
                             scored["score_risk"] +
                             scored["score_cost"]) / 3

    scored = scored.sort_values("total_score", ascending=False).reset_index(drop=True)
    scored["score_rank"] = scored.index + 1
    print(f"Stage 1 完成：有效評分 {len(scored)} 檔")
    return scored


# ══════════════════════════════════════════════════════════════
# STAGE 2：優選池（前 10 檔，債券+其他 ≥ 2）→ MVO
# ══════════════════════════════════════════════════════════════

def stage2_elite_pool(scored: pd.DataFrame) -> pd.DataFrame:
    """
    取總分前 10 檔（即前 10%）
    硬約束：debt（bond+other）至少 2 檔
    若自然前 10 已滿足 → 直接用
    若不足 → 保留前 8 名股票，再從 bond/other 中補入分數最高的 2 檔
    """
    top10 = scored.head(10).copy()
    non_stock_count = (top10["category"] != "stock").sum()

    if non_stock_count < 2:
        # 從 scored 中找 bond/other 最高分，尚未在 top10 的
        in_top10 = set(top10["ticker"])
        bond_other = scored[scored["category"] != "stock"]
        candidates = bond_other[~bond_other["ticker"].isin(in_top10)]

        need = 2 - non_stock_count
        fill = candidates.head(need)

        # 把 top10 最末的 stock 替換掉
        stock_rows = top10[top10["category"] == "stock"]
        drop_idx   = stock_rows.index[-need:]
        top10 = pd.concat([
            top10.drop(index=drop_idx),
            fill
        ]).sort_values("total_score", ascending=False).reset_index(drop=True)

    top10["pool_rank"] = top10.index + 1
    print(f"Stage 2 優選池：{len(top10)} 檔")
    print("  組成：", top10.groupby("category")["ticker"].apply(list).to_dict())
    return top10


def mvo_optimize(tickers: list, start: str, end: str,
                 n_sim: int = 5000, rf: float = 0.015) -> dict:
    """Monte Carlo MVO，最大化夏普比率"""
    tw = [f"{t}.TW" for t in tickers]
    prices = yf.download(tw, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices.columns = [c.replace(".TW", "") for c in prices.columns]
    prices = prices.ffill().dropna()

    rets = prices.pct_change().dropna()
    valid = rets.columns.tolist()
    rets  = rets[valid]
    mu    = rets.mean() * 252
    cov   = rets.cov()  * 252
    n     = len(valid)

    best_sharpe, best_w = -np.inf, None
    frontier = []

    for _ in range(n_sim):
        w    = np.random.dirichlet(np.ones(n))
        r    = float(w @ mu)
        vol  = float(np.sqrt(w @ cov.values @ w))
        shp  = (r - rf) / vol if vol > 0 else 0
        frontier.append({"ret": round(r,4), "vol": round(vol,4),
                         "sharpe": round(shp,4)})
        if shp > best_sharpe:
            best_sharpe, best_w = shp, w

    weights = {t: round(float(w), 4) for t, w in zip(valid, best_w)}
    return {
        "tickers":          valid,
        "weights":          weights,
        "expected_return":  round(float(best_w @ mu), 4),
        "volatility":       round(float(np.sqrt(best_w @ cov.values @ best_w)), 4),
        "sharpe":           round(best_sharpe, 4),
        "risk_free_rate":   rf,
        "frontier":         frontier,
    }


# ══════════════════════════════════════════════════════════════
# 回測：Smart-E 10 檔 vs. 100 檔等權
# ══════════════════════════════════════════════════════════════

def backtest(smart_weights: dict, all100_tickers: list,
             start: str, end: str) -> dict:
    # 合併下載
    all_tickers = list(set(list(smart_weights.keys()) + all100_tickers))
    tw = [f"{t}.TW" for t in all_tickers]
    prices = yf.download(tw, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices.columns = [c.replace(".TW", "") for c in prices.columns]
    prices = prices.ffill().dropna()

    def calc_perf(tickers, weights_arr):
        valid = [t for t in tickers if t in prices.columns]
        px    = prices[valid].dropna()
        w     = weights_arr[[tickers.index(t) for t in valid]]
        w     = w / w.sum()
        rets  = px.pct_change().dropna()
        port  = (rets * w).sum(axis=1)
        cum   = (1 + port).cumprod()
        roll_max = cum.cummax()
        mdd   = float(((cum - roll_max) / roll_max).min())
        shp   = float(port.mean() / port.std() * np.sqrt(252)) if port.std() > 0 else 0
        return cum, mdd, shp

    # Smart-E
    smart_t = list(smart_weights.keys())
    smart_w = np.array([smart_weights[t] for t in smart_t])
    smart_cum, smart_mdd, smart_shp = calc_perf(smart_t, smart_w)

    # 100 檔等權
    eq_t = all100_tickers
    eq_w = np.ones(len(eq_t)) / len(eq_t)
    eq_cum, eq_mdd, eq_shp = calc_perf(eq_t, eq_w)

    # 對齊日期
    common_idx = smart_cum.index.intersection(eq_cum.index)
    smart_cum  = smart_cum.reindex(common_idx)
    eq_cum     = eq_cum.reindex(common_idx)

    dates = [d.strftime("%Y-%m-%d") for d in common_idx]
    return {
        "dates":        dates,
        "smart_e":      [round(v, 4) for v in smart_cum.values],
        "equal_weight": [round(v, 4) for v in eq_cum.values],
        "metrics": {
            "smart_e": {
                "cumulative_return": round(float(smart_cum.iloc[-1] - 1), 4),
                "max_drawdown":      round(smart_mdd, 4),
                "sharpe":            round(smart_shp, 4),
            },
            "equal_weight_100": {
                "cumulative_return": round(float(eq_cum.iloc[-1] - 1), 4),
                "max_drawdown":      round(eq_mdd, 4),
                "sharpe":            round(eq_shp, 4),
            },
        },
    }


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=3*365)).strftime("%Y-%m-%d")

    print("=" * 55)
    print("Smart-E Robo Pipeline 啟動")
    print(f"資料區間：{start_date} → {end_date}")
    print("=" * 55)

    # ── Stage 0
    print("\n【Stage 0】市值預篩...")
    df100, slots = stage0_market_cap_filter()

    # ── Stage 1
    print("\n【Stage 1】量化評分...")
    scored = stage1_scoring(df100, start_date, end_date)

    # ── Stage 2
    print("\n【Stage 2】優選池 + MVO...")
    elite = stage2_elite_pool(scored)
    mvo   = mvo_optimize(elite["ticker"].tolist(), start_date, end_date)
    print(f"  MVO 夏普比率：{mvo['sharpe']}")
    print(f"  配置權重：{mvo['weights']}")

    # ── 回測
    print("\n【回測】Smart-E vs. 100 檔等權...")
    bt = backtest(mvo["weights"], df100["ticker"].tolist(), start_date, end_date)
    sm = bt["metrics"]["smart_e"]
    eq = bt["metrics"]["equal_weight_100"]
    print(f"  Smart-E    累積報酬 {sm['cumulative_return']:.1%}  "
          f"MDD {sm['max_drawdown']:.1%}  Sharpe {sm['sharpe']:.2f}")
    print(f"  Equal(100) 累積報酬 {eq['cumulative_return']:.1%}  "
          f"MDD {eq['max_drawdown']:.1%}  Sharpe {eq['sharpe']:.2f}")

    # ── 輸出 JSON
    output = {
        "generated_at": datetime.now().isoformat(),
        "period":        {"start": start_date, "end": end_date},
        "benchmarks":    BENCHMARKS,
        "stage0": {
            "slots":     slots,
            "total":     len(df100),
            "etf_list":  df100.to_dict("records"),
        },
        "stage1": {
            "scored_count": len(scored),
            "all_scores": scored[[
                "ticker","category","mktcap","mktcap_rank","cost",
                "ADV","active_risk","score_adv","score_risk","score_cost",
                "total_score","score_rank"
            ]].round(3).to_dict("records"),
        },
        "stage2": {
            "elite_pool": elite[[
                "ticker","category","mktcap","cost",
                "total_score","score_rank","pool_rank"
            ]].round(3).to_dict("records"),
            "mvo": mvo,
        },
        "backtest": bt,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n✅ 完成！結果已寫入 docs/data/results.json")


if __name__ == "__main__":
    main()
