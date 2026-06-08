#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
import requests
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/rolling_ohlcv.parquet"
DAILY_DIR = ROOT / "data/daily"
PUBLIC_DIR = ROOT / "public"
HISTORY_DIR = PUBLIC_DIR / "history"
MODELS_DIR = ROOT / "models"
UNIVERSE_PATH = ROOT / "data/universe.csv"

STRATEGY_NAME = "主板双高猎手 T2 V1 - 每日Top1版"
ALPHA = 16.0
TOP_N = 10
KEEP_TRADING_DAYS = 220
LIMIT_UP_OPEN = 0.098
FORWARD_HORIZON = 20
PEAK20_TARGET = 0.08
BROKER_COMMISSION = 0.00012
TRANSFER_FEE = 0.00001
STAMP_TAX = 0.00050
OPEN_COST = BROKER_COMMISSION + TRANSFER_FEE
CLOSE_COST = BROKER_COMMISSION + TRANSFER_FEE + STAMP_TAX
SITE_URL = os.getenv("SITE_URL", "https://a.292828.xyz")
AKSHARE_HIST_WORKERS = int(os.getenv("AKSHARE_HIST_WORKERS", "6"))
AKSHARE_HIST_RETRIES = int(os.getenv("AKSHARE_HIST_RETRIES", "3"))
AKSHARE_USE_HIST_DAILY = os.getenv("AKSHARE_USE_HIST_DAILY", "").strip().lower() in {"1", "true", "yes", "y"}

FEATURE_COLS = json.loads((MODELS_DIR / "feature_cols.json").read_text(encoding="utf-8"))

DUAL_HIGH_Q975_THRESHOLD = 0.42082220809320714
WINRATE_H4_Q950_THRESHOLD = 0.2206216747927586
WINRATE_H4_MODEL_PATH = MODELS_DIR / "win_h4.txt"
V12_MIN_AMOUNT_MA20 = 30_000_000.0
V12_MIN_CLOSE = 2.0
V12_SKIP_TOP_RANKS = 3
V12_RANGE_FILTER_RANK_LOW = 16
V12_RANGE_FILTER_RANK_HIGH = 30
V12_RANGE_FILTER_GT = 0.085
PEAK20_ALPHA = 5.0
PEAK20_THRESHOLD = 1.7010531667057818
PEAK20_HIT_MODEL_PATH = MODELS_DIR / "peak20_hit_classifier.txt"
PEAK20_PEAK_MODEL_PATH = MODELS_DIR / "peak20_peak_return_regressor.txt"

STRATEGY_DEFS: list[dict[str, Any]] = [
    {
        "id": "qlib_peak20_tp10_trail8",
        "name": "QLIB-Peak20 高收益潜力选股",
        "short_name": "Peak20主策略",
        "model_family": "qlib_peak20",
        "score_col": "peak20_score",
        "pred_win_col": "peak20_pred_hit",
        "pred_ret_col": "peak20_pred_peak",
        "metric1_label": "20日8%概率",
        "metric1_col": "peak20_pred_hit",
        "metric1_format": "pct",
        "metric2_label": "峰值收益预测",
        "metric2_col": "peak20_pred_peak",
        "metric2_format": "pct",
        "threshold": PEAK20_THRESHOLD,
        "top_n": 10,
        "profile": "peak20_alpha5_q975_tp10_or_trail8",
        "trade_rule": "T日收盘按Peak20模型打分；T+1开盘按Top10顺序检查，避开开盘涨停后买入。建议卖出规则：若收盘浮盈≥10%，次日开盘止盈；或曾浮盈≥8%且较高点回撤≥3个百分点，次日开盘卖出；跌停无法卖出则顺延。",
        "selection_note": "当前网站主策略。目标是未来20个交易日内出现较大收益机会；从2026-05-06开始纳入网站每日选股。V3动作价值模型仅作辅助参考，主卖出采用TP10或Trail8简单规则。",
        "backtest": {
            "2025": {"cum_return": 25.690861, "annual_return": None, "max_drawdown": -0.951975, "trade_win_rate": 0.670659, "trades": 167},
            "2026_ytd_to_0603": {"cum_return": 6.684822, "annual_return": None, "max_drawdown": -0.570971, "trade_win_rate": 0.681818, "trades": 66},
            "note": "回测口径来自统一模型家族对比：QLIB-Peak20买入 + TP10或Trail8次日开盘卖出。累计收益为每日等权复利研究口径，重点参考交易胜率和单笔表现。",
        },
        "reports": [
            {"label": "模型家族对比", "href": "./reports/akquant_qlib_model_comparison_report.html"},
            {"label": "V3卖出模型研究", "href": "./reports/exit_v3_action_value_report.html"},
        ],
    },
    {
        "id": "dualhigh_daily_top1_slots2",
        "name": "主板双高猎手 T2 V1 - 每日Top1版（2仓位回测）",
        "short_name": "每日Top1",
        "model_family": "dual_high",
        "score_col": "dual_score",
        "pred_win_col": "dual_pred_win",
        "pred_ret_col": "dual_pred_ret",
        "metric1_label": "胜率预测",
        "metric1_col": "dual_pred_win",
        "metric1_format": "pct",
        "metric2_label": "收益预测",
        "metric2_col": "dual_pred_ret",
        "metric2_format": "pct",
        "threshold": None,
        "top_n": 10,
        "profile": "daily_top1_no_score_threshold_slots2",
        "trade_rule": "T日收盘打分；T+1开盘按Top10顺序检查，选择第一只未涨停且可买入股票；T+2收盘前卖出，跌停则顺延。",
        "selection_note": "保证每个可交易日尝试买入1只；这里展示Top10候选，实盘按顺位避开开盘涨停。",
        "backtest": {
            "2025": {"cum_return": 0.269454, "annual_return": 0.263237, "max_drawdown": -0.088978, "trade_win_rate": 0.493776, "trades": 241},
            "2026_ytd_to_0603": {"cum_return": 0.249708, "annual_return": 0.718331, "max_drawdown": -0.100405, "trade_win_rate": 0.46875, "trades": 96},
        },
        "reports": [
            {"label": "2025回测", "href": "./reports/2025_daily_top1_alpha16_slots2_report.html"},
            {"label": "2026回测", "href": "./reports/2026_ytd_daily_top1_alpha16_slots2_report.html"},
        ],
    },
    {
        "id": "dualhigh_alpha16_q975",
        "name": "主板双高猎手 T2 V1 - 高阈值版",
        "short_name": "双高高阈值",
        "model_family": "dual_high",
        "score_col": "dual_score",
        "pred_win_col": "dual_pred_win",
        "pred_ret_col": "dual_pred_ret",
        "metric1_label": "胜率预测",
        "metric1_col": "dual_pred_win",
        "metric1_format": "pct",
        "metric2_label": "收益预测",
        "metric2_col": "dual_pred_ret",
        "metric2_format": "pct",
        "threshold": DUAL_HIGH_Q975_THRESHOLD,
        "top_n": 10,
        "profile": "dualhigh_alpha16_q975",
        "trade_rule": "T日收盘打分；只有综合分超过训练期97.5%阈值才入选；T+1开盘买入，T+2收盘前卖出，跌停顺延。",
        "selection_note": "高胜率/高收益阈值版，交易次数更少；若当天无股票过阈值则显示空表。",
        "backtest": {
            "2025": {"cum_return": 0.191757, "annual_return": 0.187463, "max_drawdown": -0.106628, "trade_win_rate": 0.54386, "trades": 57},
            "2026_ytd_to_0603": {"cum_return": 0.013726, "annual_return": 0.033662, "max_drawdown": -0.023247, "trade_win_rate": 0.777778, "trades": 9},
        },
        "reports": [
            {"label": "2025回测", "href": "./reports/2025_dualhigh_alpha16_q975_report.html"},
            {"label": "2026回测", "href": "./reports/2026_ytd_dualhigh_alpha16_q975_report.html"},
        ],
    },
    {
        "id": "high_winrate_h4_q950",
        "name": "主板胜率猎手 V1 - H4高胜率版",
        "short_name": "胜率猎手H4",
        "model_family": "winrate_h4",
        "score_col": "pred_win_h4",
        "pred_win_col": "pred_win_h4",
        "pred_ret_col": None,
        "metric1_label": "胜率预测",
        "metric1_col": "pred_win_h4",
        "metric1_format": "pct",
        "metric2_label": "收益预测",
        "metric2_col": None,
        "metric2_format": "pct",
        "threshold": WINRATE_H4_Q950_THRESHOLD,
        "top_n": 10,
        "profile": "h4_q950",
        "trade_rule": "T日收盘打分；T+1开盘买入；原始版本目标T+4收盘卖出，按胜率模型筛选。",
        "selection_note": "最早训练的高胜率策略，偏向胜率和稳定性，展示超过95%训练期阈值的候选。",
        "backtest": {
            "2025": {"cum_return": 0.229011, "annual_return": 0.221818, "max_drawdown": -0.098581, "trade_win_rate": 0.647059, "trades": 51},
            "2026_ytd_to_0603": {"cum_return": 0.20353, "annual_return": 0.568179, "max_drawdown": -0.150267, "trade_win_rate": 0.703704, "trades": 27},
        },
        "reports": [
            {"label": "2025回测", "href": "./reports/2025_high_winrate_h4_q950_report.html"},
            {"label": "2026回测", "href": "./reports/2026_ytd_high_winrate_h4_q950_report.html"},
        ],
    },
    {
        "id": "v12_close_only",
        "name": "V1.2 收盘选股版",
        "short_name": "V1.2收盘",
        "model_family": "rule_v12_close",
        "score_col": "v12_score",
        "pred_win_col": None,
        "pred_ret_col": None,
        "metric1_label": "V03A排名",
        "metric1_col": "v12_rank_v03a",
        "metric1_format": "int",
        "metric2_label": "日振幅",
        "metric2_col": "daily_range",
        "metric2_format": "pct",
        "threshold": None,
        "top_n": 15,
        "profile": "v12_close_only_no_open_gap",
        "custom_filter": "v12_close_only",
        "trade_rule": "T日收盘计算V03A分数并直接选股；取消原V1.2的T+1开盘gap择机触发；后续可按T+1开盘买入、T+2收盘卖出观察。",
        "selection_note": "由原V1.2事件策略改造为收盘选股版：保留V03A打分、跳过前3名、成交额/价格过滤和振幅排除规则；不再等待开盘gap确认。",
        "backtest": {
            "2026_ytd_to_0603": {"cum_return": 0.575945, "annual_return": 2.22079, "max_drawdown": -0.110579, "trade_win_rate": 0.491132, "trades": 4398},
            "note": "指标来自原V1.2事件策略回测；收盘选股版取消开盘gap后尚未单独重跑回测，仅作规则来源参考。",
        },
        "reports": [],
    },
]


def now_cn() -> datetime:
    return datetime.now(pytz.timezone("Asia/Shanghai"))


def today_str_cn() -> str:
    return now_cn().strftime("%Y-%m-%d")


def ensure_dirs() -> None:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def load_rolling() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"missing rolling data: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def is_cn_trade_day(date_str: str) -> bool:
    """Use AkShare trade calendar. If calendar fetch fails, run on weekdays only."""
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        ds = pd.to_datetime(cal["trade_date"]).dt.strftime("%Y-%m-%d")
        return date_str in set(ds)
    except Exception as exc:
        print(f"[WARN] trade calendar unavailable, fallback weekday check: {exc}")
        return pd.Timestamp(date_str).weekday() < 5


def fetch_akshare_spot(date_str: str, rolling: pd.DataFrame) -> pd.DataFrame:
    import akshare as ak

    print("[INFO] fetching AkShare stock_zh_a_spot_em ...")
    raw = ak.stock_zh_a_spot_em()
    if raw.empty:
        raise RuntimeError("AkShare returned empty spot data")

    col = {c: c for c in raw.columns}
    required = ["代码", "名称", "最新价", "最高", "最低", "今开", "昨收", "成交量", "成交额", "换手率"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"AkShare columns missing: {missing}; got {list(raw.columns)}")

    df = raw.rename(
        columns={
            "代码": "code",
            "名称": "name",
            "最新价": "raw_close",
            "最高": "raw_high",
            "最低": "raw_low",
            "今开": "raw_open",
            "昨收": "raw_prev_close",
            "成交量": "raw_volume_lot",
            "成交额": "amount",
            "换手率": "turnover_pct",
        }
    )
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].str.startswith(("00", "60"))].copy()
    df = df[~df["name"].fillna("").astype(str).str.contains("ST|退", case=False, regex=True)].copy()
    for c in ["raw_close", "raw_high", "raw_low", "raw_open", "raw_prev_close", "raw_volume_lot", "amount", "turnover_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["raw_close", "raw_high", "raw_low", "raw_open", "raw_prev_close"])
    df = df[(df["raw_close"] > 0) & (df["raw_open"] > 0) & (df["amount"] > 0)].copy()

    latest = rolling.sort_values("date").groupby("code", as_index=False).tail(1)[["code", "close"]].rename(columns={"close": "adj_prev_close"})
    df = df.merge(latest, on="code", how="left")
    # AkShare spot is raw/unadjusted while the seed history is Qlib-adjusted.
    # Make today's OHLC continuous with the rolling adjusted series.
    df["scale"] = df["adj_prev_close"] / df["raw_prev_close"]
    df.loc[~np.isfinite(df["scale"]) | (df["scale"] <= 0), "scale"] = 1.0
    for raw_c, out_c in [("raw_open", "open"), ("raw_high", "high"), ("raw_low", "low"), ("raw_close", "close")]:
        df[out_c] = df[raw_c] * df["scale"]
    # Eastmoney spot volume is in lots. Qlib seed is shares.
    df["volume"] = df["raw_volume_lot"] * 100.0
    # Qlib turnover is fraction, Eastmoney turnover is percent.
    df["turnover"] = df["turnover_pct"] / 100.0
    df["date"] = pd.Timestamp(date_str).normalize()
    out = df[["code", "name", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "raw_open", "raw_high", "raw_low", "raw_close", "raw_prev_close", "scale"]].copy()
    return out.sort_values("code").reset_index(drop=True)


def code_to_sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"sh{code}"
    return f"sz{code}"


def historical_fetch_universe(rolling: pd.DataFrame) -> pd.DataFrame:
    names = load_universe_names()
    if names.empty:
        names = pd.DataFrame({"code": sorted(rolling["code"].astype(str).str.zfill(6).unique()), "name": ""})
    names["code"] = names["code"].astype(str).str.zfill(6)
    names = names[names["code"].str.startswith(("00", "60"))].drop_duplicates("code").sort_values("code").reset_index(drop=True)
    return names


def _fetch_one_sina_daily(code: str, start_yyyymmdd: str, end_yyyymmdd: str, target_ts: pd.Timestamp) -> dict[str, Any] | None:
    """Fetch one stock's raw daily bar from AkShare/Sina.

    This endpoint is slower than Eastmoney spot but supports historical dates, which is
    required when a GitHub Actions manual run backfills a past trading day.
    """
    import akshare as ak

    symbol = code_to_sina_symbol(code)
    last_exc: Exception | None = None
    for attempt in range(AKSHARE_HIST_RETRIES):
        try:
            raw = ak.stock_zh_a_daily(symbol=symbol, start_date=start_yyyymmdd, end_date=end_yyyymmdd, adjust="")
            if raw is None or raw.empty:
                return None
            df = raw.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.sort_values("date")
            target_rows = df[df["date"].eq(target_ts)]
            if target_rows.empty:
                return None
            target = target_rows.iloc[-1]
            prev_rows = df[df["date"].lt(target_ts)]
            prev = prev_rows.iloc[-1] if not prev_rows.empty else None
            return {
                "code": str(code).zfill(6),
                "date": target_ts,
                "raw_open": target.get("open"),
                "raw_high": target.get("high"),
                "raw_low": target.get("low"),
                "raw_close": target.get("close"),
                "volume": target.get("volume"),
                "amount": target.get("amount"),
                # AkShare/Sina turnover is already a fraction, not a percent.
                "turnover": target.get("turnover"),
                "raw_prev_date": prev.get("date") if prev is not None else pd.NaT,
                "raw_prev_close": prev.get("close") if prev is not None else np.nan,
            }
        except Exception as exc:  # pragma: no cover - network defensive path
            last_exc = exc
            if attempt + 1 < AKSHARE_HIST_RETRIES:
                time.sleep(0.35 * (attempt + 1))
    print(f"[WARN] historical fetch failed for {code}: {last_exc}")
    return None


def _fetch_sina_daily_chunk(codes: list[str], start_yyyymmdd: str, end_yyyymmdd: str, date_str: str) -> list[dict[str, Any]]:
    target_ts = pd.Timestamp(date_str).normalize()
    out: list[dict[str, Any]] = []
    for code in codes:
        row = _fetch_one_sina_daily(code, start_yyyymmdd, end_yyyymmdd, target_ts)
        if row is not None:
            out.append(row)
    return out


def fetch_akshare_hist_date(date_str: str, rolling: pd.DataFrame) -> pd.DataFrame:
    """Fetch a historical daily bar for all mainboard names via AkShare/Sina."""
    target_ts = pd.Timestamp(date_str).normalize()
    # Pull a small lookback window so suspended stocks can still find their most
    # recent raw previous close for adjustment continuity.
    start_yyyymmdd = (target_ts - pd.Timedelta(days=30)).strftime("%Y%m%d")
    end_yyyymmdd = target_ts.strftime("%Y%m%d")
    universe = historical_fetch_universe(rolling)
    codes = universe["code"].tolist()
    print(f"[INFO] fetching AkShare stock_zh_a_daily for {date_str}: {len(codes)} symbols, workers={AKSHARE_HIST_WORKERS}")

    rows: list[dict[str, Any]] = []
    workers = max(1, min(AKSHARE_HIST_WORKERS, len(codes)))
    if workers == 1:
        rows = _fetch_sina_daily_chunk(codes, start_yyyymmdd, end_yyyymmdd, date_str)
    else:
        # AkShare's Sina decoder uses MiniRacer, which can crash when called from
        # multiple threads. Use multiple processes and keep each process sequential.
        chunks = [codes[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_sizes = {
                executor.submit(_fetch_sina_daily_chunk, chunk, start_yyyymmdd, end_yyyymmdd, date_str): len(chunk)
                for chunk in chunks
            }
            done_symbols = 0
            for future in as_completed(future_sizes):
                chunk_rows = future.result()
                rows.extend(chunk_rows)
                done_symbols += future_sizes[future]
                print(f"[INFO] historical fetch progress {done_symbols}/{len(codes)}, rows={len(rows)}")

    if not rows:
        raise RuntimeError(f"AkShare historical fetch returned no rows for {date_str}")

    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["raw_prev_date"] = pd.to_datetime(df["raw_prev_date"], errors="coerce").dt.normalize()
    numeric_cols = ["raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount", "turnover", "raw_prev_close"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["raw_open", "raw_high", "raw_low", "raw_close", "volume", "amount"])
    df = df[(df["raw_open"] > 0) & (df["raw_close"] > 0) & (df["amount"] > 0)].copy()

    prev_adj = rolling[["code", "date", "close"]].copy()
    prev_adj["code"] = prev_adj["code"].astype(str).str.zfill(6)
    prev_adj["date"] = pd.to_datetime(prev_adj["date"]).dt.normalize()
    prev_adj = prev_adj.rename(columns={"date": "raw_prev_date", "close": "adj_prev_close"})
    df = df.merge(prev_adj, on=["code", "raw_prev_date"], how="left")

    latest_before = (
        rolling[rolling["date"].lt(target_ts)]
        .sort_values(["code", "date"])
        .groupby("code", as_index=False)
        .tail(1)[["code", "date", "close"]]
        .rename(columns={"date": "fallback_prev_date", "close": "fallback_adj_prev_close"})
    )
    latest_before["code"] = latest_before["code"].astype(str).str.zfill(6)
    df = df.merge(latest_before, on="code", how="left")
    df["adj_prev_close"] = df["adj_prev_close"].fillna(df["fallback_adj_prev_close"])

    df["scale"] = df["adj_prev_close"] / df["raw_prev_close"]
    df.loc[~np.isfinite(df["scale"]) | (df["scale"] <= 0), "scale"] = 1.0
    for raw_c, out_c in [("raw_open", "open"), ("raw_high", "high"), ("raw_low", "low"), ("raw_close", "close")]:
        df[out_c] = df[raw_c] * df["scale"]

    df = df.merge(universe, on="code", how="left")
    out = df[
        [
            "code", "name", "date", "open", "high", "low", "close", "volume", "amount", "turnover",
            "raw_open", "raw_high", "raw_low", "raw_close", "raw_prev_date", "raw_prev_close", "scale",
        ]
    ].copy()
    print(f"[INFO] historical rows usable for {date_str}: {len(out)}")
    return out.sort_values("code").reset_index(drop=True)


def fetch_akshare_for_date(date_str: str, rolling: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Fetch one trading day.

    For the current date we try the faster Eastmoney spot endpoint. For backfills
    (for example, manually pulling 2026-06-04 after the site still shows
    2026-06-03), use AkShare's Sina daily endpoint because spot has no date
    parameter.
    """
    if AKSHARE_USE_HIST_DAILY:
        print("[INFO] AKSHARE_USE_HIST_DAILY enabled; use historical daily endpoint instead of realtime spot")
        return fetch_akshare_hist_date(date_str, rolling), "AkShare stock_zh_a_daily"
    if date_str == today_str_cn():
        try:
            return fetch_akshare_spot(date_str, rolling), "AkShare stock_zh_a_spot_em"
        except Exception as exc:
            print(f"[WARN] spot fetch failed, fallback to historical daily endpoint: {exc}")
    return fetch_akshare_hist_date(date_str, rolling), "AkShare stock_zh_a_daily"


def append_today(rolling: pd.DataFrame, today: pd.DataFrame) -> pd.DataFrame:
    save_daily = today.copy()
    save_daily.to_csv(DAILY_DIR / f"{today['date'].iloc[0].date()}.csv", index=False)
    append_cols = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    merged = pd.concat([rolling[append_cols], today[append_cols]], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    merged = merged.drop_duplicates(["code", "date"], keep="last")
    keep_dates = sorted(merged["date"].unique())[-KEEP_TRADING_DAYS:]
    merged = merged[merged["date"].isin(keep_dates)].sort_values(["code", "date"]).reset_index(drop=True)
    merged.to_parquet(DATA_PATH, index=False)
    return merged


def compute_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy().sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", group_keys=False)
    df["prev_close"] = g["close"].shift(1)
    df["ret_1"] = df["close"] / df["prev_close"] - 1.0
    df["intraday_return"] = df["close"] / df["open"] - 1.0
    df["gap_open"] = df["open"] / df["prev_close"] - 1.0
    df["daily_range"] = df["high"] / df["low"] - 1.0
    range_den = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_position"] = ((df["close"] - df["low"]) / range_den).clip(0.0, 1.0).fillna(0.5)
    df["upper_shadow"] = df["high"] / np.maximum(df["open"], df["close"]) - 1.0
    df["lower_shadow"] = np.minimum(df["open"], df["close"]) / df["low"] - 1.0
    df["body"] = df["close"] / df["open"] - 1.0

    for w in [3, 5, 10, 20, 60]:
        df[f"mom_{w}"] = g["close"].pct_change(w)
    for w in [5, 10, 20, 60]:
        df[f"volatility_{w}"] = g["ret_1"].transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
    for w in [5, 10, 20]:
        df[f"amplitude_{w}"] = g["daily_range"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
    for w in [5, 10, 20, 60]:
        ma = g["close"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        df[f"ma{w}"] = ma
        df[f"close_ma{w}_bias"] = df["close"] / ma - 1.0
    df["ma5_over_ma20"] = df["ma5"] / df["ma20"] - 1.0
    df["ma20_over_ma60"] = df["ma20"] / df["ma60"] - 1.0
    df["max_ret_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=20).max())
    df["min_ret_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=20).min())
    rolling_high20 = g["high"].transform(lambda s: s.rolling(20, min_periods=20).max())
    rolling_low20 = g["low"].transform(lambda s: s.rolling(20, min_periods=20).min())
    df["drawdown_20"] = df["close"] / rolling_high20 - 1.0
    df["up_from_low20"] = df["close"] / rolling_low20 - 1.0

    for base in ["volume", "amount", "turnover"]:
        for w in [5, 20, 60]:
            ma = g[base].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
            df[f"{base}_ma{w}"] = ma
            df[f"{base}_ratio_{w}"] = df[base] / (ma + 1e-12)
        mean20 = g[base].transform(lambda s: s.rolling(20, min_periods=20).mean())
        std20 = g[base].transform(lambda s: s.rolling(20, min_periods=20).std())
        df[f"{base}_z20"] = (df[base] - mean20) / (std20 + 1e-12)

    df["illiquidity_20"] = (
        df["ret_1"].abs() / (df["amount"] / 100_000_000.0 + 1e-12)
    ).groupby(df["code"]).transform(lambda s: s.rolling(20, min_periods=20).mean())

    df["amount_ratio_20_clip"] = df["amount_ratio_20"].clip(upper=3.0)
    df["turnover_ratio_20_clip"] = df["turnover_ratio_20"].clip(upper=3.0)
    df["amount_extreme"] = (df["amount_ratio_20_clip"] - 1.5).abs()
    df["turnover_extreme"] = (df["turnover_ratio_20_clip"] - 1.5).abs()

    dg = df.groupby("date", sort=False)
    df["score_v03a_like"] = (
        0.25 * dg["upper_shadow"].rank(pct=True, ascending=True)
        + 0.20 * dg["close_position"].rank(pct=True, ascending=False)
        + 0.20 * dg["intraday_return"].rank(pct=True, ascending=False)
        + 0.15 * dg["volatility_10"].rank(pct=True, ascending=True)
        + 0.10 * dg["amount_extreme"].rank(pct=True, ascending=True)
        + 0.10 * dg["turnover_extreme"].rank(pct=True, ascending=True)
    )
    df["rank_v03a_like"] = dg["score_v03a_like"].rank(method="first", pct=True, ascending=True)

    rank_base_cols = [
        "mom_5", "mom_20", "mom_60", "volatility_10", "volatility_20", "daily_range", "close_position",
        "upper_shadow", "lower_shadow", "amount_ratio_20", "turnover_ratio_20", "amount_z20", "turnover_z20",
        "illiquidity_20", "score_v03a_like",
    ]
    for c in rank_base_cols:
        df[f"cs_rank_{c}"] = df.groupby("date")[c].rank(pct=True, method="average")
    return df


def load_universe_names() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        return pd.DataFrame(columns=["code", "name"])
    u = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    u["code"] = u["code"].astype(str).str.zfill(6)
    return u[["code", "name"]].drop_duplicates("code")


def load_raw_prices(date_str: str) -> pd.DataFrame:
    """Load raw/unadjusted prices that match market software display prices."""
    path = DAILY_DIR / f"{date_str}.csv"
    cols = ["code", "raw_open", "raw_high", "raw_low", "raw_close", "raw_prev_close"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].drop_duplicates("code")


def load_raw_close_for_date(date_str: str) -> pd.DataFrame:
    """Return raw close for a date from same-day file or next-day raw_prev_close."""
    direct = load_raw_prices(date_str)
    if not direct.empty and direct["raw_close"].notna().any():
        return direct[["code", "raw_open", "raw_high", "raw_low", "raw_close"]].copy()

    frames: list[pd.DataFrame] = []
    for path in sorted(DAILY_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, dtype={"code": str})
        except Exception:
            continue
        if "raw_prev_date" not in df.columns or "raw_prev_close" not in df.columns:
            continue
        df["raw_prev_date"] = pd.to_datetime(df["raw_prev_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        m = df[df["raw_prev_date"].eq(date_str)].copy()
        if m.empty:
            continue
        m["code"] = m["code"].astype(str).str.zfill(6)
        m["raw_close"] = pd.to_numeric(m["raw_prev_close"], errors="coerce")
        m["raw_open"] = np.nan
        m["raw_high"] = np.nan
        m["raw_low"] = np.nan
        frames.append(m[["code", "raw_open", "raw_high", "raw_low", "raw_close"]])
    if not frames:
        return pd.DataFrame(columns=["code", "raw_open", "raw_high", "raw_low", "raw_close"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["raw_close"]).drop_duplicates("code", keep="last")


def payload_item_groups(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    item_groups: list[list[dict[str, Any]]] = []
    if isinstance(payload.get("items"), list):
        item_groups.append(payload["items"])
    for strategy in payload.get("strategies", []) or []:
        if isinstance(strategy.get("items"), list):
            item_groups.append(strategy["items"])
    return item_groups


def trade_net_return(exit_price: Any, buy_price: Any) -> float | None:
    buy = finite_float(buy_price)
    sell = finite_float(exit_price)
    if buy is None or sell is None or buy <= 0 or sell <= 0:
        return None
    return sell * (1.0 - CLOSE_COST) / (buy * (1.0 + OPEN_COST)) - 1.0


def score_latest(
    panel: pd.DataFrame,
    names: pd.DataFrame | None = None,
    target_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    latest_date = pd.Timestamp(target_date).normalize() if target_date is not None else panel["date"].max()
    latest = panel[panel["date"].eq(latest_date)].dropna(subset=FEATURE_COLS).copy()
    if latest.empty:
        raise RuntimeError(f"no scorable rows for {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}")
    latest = latest[latest["code"].str.startswith(("00", "60"))]
    latest = latest[(latest["open"] > 0) & (latest["close"] > 0) & (latest["amount"] > 0)]
    win_model = lgb.Booster(model_file=str(MODELS_DIR / "win_classifier.txt"))
    ret_model = lgb.Booster(model_file=str(MODELS_DIR / "return_regressor.txt"))
    x = latest[FEATURE_COLS]
    latest["dual_pred_win"] = win_model.predict(x)
    latest["dual_pred_ret"] = ret_model.predict(x)
    latest["dual_score"] = latest["dual_pred_win"] + ALPHA * latest["dual_pred_ret"].clip(-0.05, 0.12)
    # Backward-compatible aliases used by the primary strategy and older JSON.
    latest["pred_win"] = latest["dual_pred_win"]
    latest["pred_ret"] = latest["dual_pred_ret"]
    latest["score"] = latest["dual_score"]

    if WINRATE_H4_MODEL_PATH.exists():
        win_h4_model = lgb.Booster(model_file=str(WINRATE_H4_MODEL_PATH))
        latest["pred_win_h4"] = win_h4_model.predict(x)
    else:
        print(f"[WARN] optional high-winrate model missing: {WINRATE_H4_MODEL_PATH}")
        latest["pred_win_h4"] = np.nan

    if PEAK20_HIT_MODEL_PATH.exists() and PEAK20_PEAK_MODEL_PATH.exists():
        peak20_hit_model = lgb.Booster(model_file=str(PEAK20_HIT_MODEL_PATH))
        peak20_peak_model = lgb.Booster(model_file=str(PEAK20_PEAK_MODEL_PATH))
        latest["peak20_pred_hit"] = peak20_hit_model.predict(x)
        latest["peak20_pred_peak"] = peak20_peak_model.predict(x)
        latest["peak20_score"] = latest["peak20_pred_hit"] + PEAK20_ALPHA * latest["peak20_pred_peak"].clip(-0.05, 0.30)
    else:
        print(f"[WARN] optional QLIB-Peak20 models missing: {PEAK20_HIT_MODEL_PATH}, {PEAK20_PEAK_MODEL_PATH}")
        latest["peak20_pred_hit"] = np.nan
        latest["peak20_pred_peak"] = np.nan
        latest["peak20_score"] = np.nan

    # V1.2 close-only rule strategy: use the deterministic V03A score computed
    # from same-day OHLCV.  Original V1.2 was an event strategy with T+1 open-gap
    # entry timing; this close-only adaptation deliberately removes the open-gap
    # trigger so it can be shown as an end-of-day stock-picking list.
    latest["v12_score"] = latest["score_v03a_like"]
    latest["v12_rank_v03a"] = latest["v12_score"].rank(method="first", ascending=False).astype(int)
    latest["v12_range_excluded"] = latest["v12_rank_v03a"].between(
        V12_RANGE_FILTER_RANK_LOW, V12_RANGE_FILTER_RANK_HIGH, inclusive="both"
    ) & (latest["daily_range"] > V12_RANGE_FILTER_GT)
    latest["v12_close_only_pass"] = (
        (latest["v12_rank_v03a"] > V12_SKIP_TOP_RANKS)
        & (~latest["v12_range_excluded"])
        & (latest["amount_ma20"] >= V12_MIN_AMOUNT_MA20)
        & (latest["close"] >= V12_MIN_CLOSE)
        & (latest["volume"] > 0)
        & (latest["amount"] > 0)
    )

    latest["rank"] = latest["dual_score"].rank(method="first", ascending=False).astype(int)
    fallback_names = load_universe_names()
    if names is not None and not names.empty:
        merged_names = pd.concat([fallback_names, names[["code", "name"]]], ignore_index=True).drop_duplicates("code", keep="last")
    else:
        merged_names = fallback_names
    if not merged_names.empty:
        latest = latest.merge(merged_names, on="code", how="left")
    else:
        latest["name"] = ""
    latest_date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    raw_prices = load_raw_prices(latest_date_str)
    if not raw_prices.empty:
        latest = latest.merge(raw_prices, on="code", how="left")
    else:
        for c in ["raw_open", "raw_high", "raw_low", "raw_close", "raw_prev_close"]:
            latest[c] = np.nan
    return latest.sort_values("dual_score", ascending=False).reset_index(drop=True)


def enrich_payload_forward_returns(payload: dict[str, Any], rolling: pd.DataFrame) -> bool:
    """Add next-trading-day executable one-day returns to a public payload.

    The stock list is produced after the T-day close, so a realistic one-day
    holding result should use T+1 open as the buy price and T+1 close as the
    mark/sell price.  The old close-to-close signal move is still retained as
    `next_1d_close_to_close_return` for audit/comparison.
    """
    if not payload.get("date"):
        return False
    signal_date = pd.Timestamp(payload["date"]).normalize()
    item_groups = payload_item_groups(payload)
    if not item_groups:
        return False

    dates = sorted(pd.to_datetime(rolling["date"]).dt.normalize().unique())
    next_dates = [pd.Timestamp(d).normalize() for d in dates if pd.Timestamp(d).normalize() > signal_date]
    if not next_dates:
        changed = False
        defaults = {
            "next_1d_date": None,
            "next_1d_return": None,
            "next_1d_close_to_close_return": None,
            "next_1d_open_gap": None,
            "next_1d_buyable": None,
            "forward20_horizon": FORWARD_HORIZON,
            "forward20_target": PEAK20_TARGET,
            "forward20_available_days": 0,
            "forward20_complete": False,
            "forward20_path": [],
        }
        for items in item_groups:
            for item in items:
                for k, v in defaults.items():
                    if item.get(k) != v:
                        item[k] = v
                        changed = True
        if changed:
            payload["performance_updated_at"] = now_cn().isoformat()
        return changed

    next_date = next_dates[0]
    next_date_str = next_date.strftime("%Y-%m-%d")
    signal_close = rolling[rolling["date"].eq(signal_date)].set_index("code")["close"]
    next_bar = rolling[rolling["date"].eq(next_date)].set_index("code")[["open", "close"]]
    next_raw = load_raw_prices(next_date_str)
    next_raw_map = next_raw.set_index("code").to_dict(orient="index") if not next_raw.empty else {}
    forward_dates = next_dates[:FORWARD_HORIZON]
    forward_complete = len(forward_dates) >= FORWARD_HORIZON
    forward_panel = rolling[rolling["date"].isin(forward_dates)][
        ["code", "date", "open", "high", "low", "close", "amount", "turnover"]
    ].copy()
    forward_by_code = {
        str(code).zfill(6): g.sort_values("date").head(FORWARD_HORIZON).reset_index(drop=True)
        for code, g in forward_panel.groupby("code", sort=False)
    }
    forward_raw_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for d in forward_dates:
        dstr = pd.Timestamp(d).strftime("%Y-%m-%d")
        raw_d = load_raw_prices(dstr)
        forward_raw_maps[dstr] = raw_d.set_index("code").to_dict(orient="index") if not raw_d.empty else {}

    changed = False
    for items in item_groups:
        for item in items:
            code = str(item.get("code", "")).zfill(6)
            base = signal_close.get(code, item.get("adjusted_close", item.get("close")))
            next_open = next_bar["open"].get(code, np.nan) if not next_bar.empty else np.nan
            next_close = next_bar["close"].get(code, np.nan) if not next_bar.empty else np.nan
            raw_info = next_raw_map.get(code, {})
            next_raw_open = finite_float(raw_info.get("raw_open"))
            next_raw_high = finite_float(raw_info.get("raw_high"))
            next_raw_low = finite_float(raw_info.get("raw_low"))
            next_raw_close = finite_float(raw_info.get("raw_close"))

            if pd.notna(base) and pd.notna(next_close) and float(base) > 0:
                close_to_close = round(float(next_close) / float(base) - 1.0, 6)
                open_gap = None
                trade_return = None
                buyable = None
                if pd.notna(next_open) and float(next_open) > 0:
                    trade_return = round(float(next_close) / float(next_open) - 1.0, 6)
                    open_gap = round(float(next_open) / float(base) - 1.0, 6)
                    # A-share limit-up rules vary for special boards/ST; this
                    # mainboard site uses the same 9.8% conservative threshold
                    # as the picker to identify likely open-limit-up, not buyable.
                    buyable = bool(open_gap < LIMIT_UP_OPEN)

                updates = {
                    "next_1d_date": next_date_str,
                    # Executable one-day result: T+1 open buy -> T+1 close.
                    "next_1d_return": trade_return,
                    "next_1d_return_basis": "next_open_to_next_close",
                    # Audit-only signal move: T close -> T+1 close.
                    "next_1d_close_to_close_return": close_to_close,
                    "next_1d_open_gap": open_gap,
                    "next_1d_buyable": buyable,
                }
                if next_raw_open is not None:
                    updates["next_1d_open"] = round(next_raw_open, 4)
                if next_raw_high is not None:
                    updates["next_1d_high"] = round(next_raw_high, 4)
                if next_raw_low is not None:
                    updates["next_1d_low"] = round(next_raw_low, 4)
                if next_raw_close is not None:
                    updates["next_1d_close"] = round(next_raw_close, 4)

                if any(item.get(k) != v for k, v in updates.items()):
                    changed = True
                item.update(updates)
            else:
                item["next_1d_date"] = next_date_str
                item["next_1d_return"] = None
                item["next_1d_close_to_close_return"] = None
                item["next_1d_open_gap"] = None
                item["next_1d_buyable"] = None

            updates20: dict[str, Any] = {
                "forward20_horizon": FORWARD_HORIZON,
                "forward20_target": PEAK20_TARGET,
                "forward20_available_days": 0,
                "forward20_complete": False,
                "forward20_return_basis": "T+1 open buy; future max close/end close; net return deducts commission, transfer fee and stamp tax",
                "forward20_path": [],
                "forward20_hit8": None,
                "forward20_grade": None,
            }
            g20 = forward_by_code.get(code)
            if (
                g20 is not None
                and len(g20) >= 1
                and pd.notna(base)
                and float(base) > 0
                and pd.notna(g20.loc[0, "open"])
                and float(g20.loc[0, "open"]) > 0
            ):
                buy_open_adj = float(g20.loc[0, "open"])
                buy_gap20 = buy_open_adj / float(base) - 1.0
                path: list[dict[str, Any]] = []
                for h, rr in enumerate(g20.itertuples(index=False), start=1):
                    dstr = pd.Timestamp(rr.date).strftime("%Y-%m-%d")
                    raw_info = forward_raw_maps.get(dstr, {}).get(code, {})
                    raw_open = finite_float(raw_info.get("raw_open"))
                    raw_high = finite_float(raw_info.get("raw_high"))
                    raw_low = finite_float(raw_info.get("raw_low"))
                    raw_close = finite_float(raw_info.get("raw_close"))
                    prev_close = float(base) if h == 1 else float(g20.loc[h - 2, "close"])
                    daily_ret = float(rr.close) / prev_close - 1.0 if pd.notna(rr.close) and prev_close > 0 else np.nan
                    close_ret = float(rr.close) / buy_open_adj - 1.0 if pd.notna(rr.close) else np.nan
                    high_ret = float(rr.high) / buy_open_adj - 1.0 if pd.notna(rr.high) else np.nan
                    low_ret = float(rr.low) / buy_open_adj - 1.0 if pd.notna(rr.low) else np.nan
                    close_net = trade_net_return(rr.close, buy_open_adj)
                    path.append(
                        {
                            "h": h,
                            "date": dstr,
                            "raw_open": round(raw_open, 4) if raw_open is not None else None,
                            "raw_high": round(raw_high, 4) if raw_high is not None else None,
                            "raw_low": round(raw_low, 4) if raw_low is not None else None,
                            "raw_close": round(raw_close, 4) if raw_close is not None else None,
                            "daily_return": round(float(daily_ret), 6) if np.isfinite(daily_ret) else None,
                            "close_return": round(float(close_ret), 6) if np.isfinite(close_ret) else None,
                            "close_net_return": round(float(close_net), 6) if close_net is not None else None,
                            "high_return": round(float(high_ret), 6) if np.isfinite(high_ret) else None,
                            "low_return": round(float(low_ret), 6) if np.isfinite(low_ret) else None,
                            "amount": round(float(rr.amount), 2) if pd.notna(rr.amount) else None,
                            "turnover": round(float(rr.turnover), 6) if pd.notna(rr.turnover) else None,
                        }
                    )

                valid_close_net = [x for x in path if finite_float(x.get("close_net_return")) is not None]
                valid_close_ret = [x for x in path if finite_float(x.get("close_return")) is not None]
                if valid_close_net:
                    peak = max(valid_close_net, key=lambda x: finite_float(x.get("close_net_return")) or -999.0)
                    trough = min(valid_close_net, key=lambda x: finite_float(x.get("close_net_return")) or 999.0)
                    end = path[-1]
                    buy_raw = finite_float(forward_raw_maps.get(path[0]["date"], {}).get(code, {}).get("raw_open"))
                    end_raw_close = finite_float(end.get("raw_close"))
                    peak_raw_close = finite_float(peak.get("raw_close"))
                    complete_item20 = len(path) >= FORWARD_HORIZON
                    peak_net_value = finite_float(peak.get("close_net_return")) or -999.0
                    updates20.update(
                        {
                            "forward20_available_days": int(len(path)),
                            "forward20_complete": bool(complete_item20),
                            "forward20_start_date": path[0]["date"],
                            "forward20_end_date": path[-1]["date"],
                            "forward20_buy_date": path[0]["date"],
                            "forward20_buy_open": round(buy_raw, 4) if buy_raw is not None else None,
                            "forward20_buy_open_adjusted": round(buy_open_adj, 4),
                            "forward20_buy_gap": round(float(buy_gap20), 6),
                            "forward20_buyable": bool(buy_gap20 < LIMIT_UP_OPEN),
                            "forward20_peak_date": peak["date"],
                            "forward20_peak_holding_day": int(peak["h"]),
                            "forward20_peak_close": round(peak_raw_close, 4) if peak_raw_close is not None else None,
                            "forward20_peak_return": round(float(finite_float(peak.get("close_return")) or 0.0), 6),
                            "forward20_peak_net_return": round(float(finite_float(peak.get("close_net_return")) or 0.0), 6),
                            "forward20_end_date_actual": end["date"],
                            "forward20_end_close": round(end_raw_close, 4) if end_raw_close is not None else None,
                            "forward20_end_return": round(float(finite_float(end.get("close_return")) or 0.0), 6),
                            "forward20_end_net_return": round(float(finite_float(end.get("close_net_return")) or 0.0), 6),
                            "forward20_min_date": trough["date"],
                            "forward20_min_net_return": round(float(finite_float(trough.get("close_net_return")) or 0.0), 6),
                            "forward20_hit8": bool(peak_net_value >= PEAK20_TARGET) if complete_item20 else None,
                            "forward20_hit8_so_far": bool(peak_net_value >= PEAK20_TARGET),
                            "forward20_grade": (
                                (
                                    "S" if peak_net_value >= 0.20
                                    else "A" if peak_net_value >= PEAK20_TARGET
                                    else "B" if peak_net_value > 0
                                    else "C"
                                )
                                if complete_item20
                                else None
                            ),
                            "forward20_grade_so_far": (
                                "S" if peak_net_value >= 0.20
                                else "A" if peak_net_value >= PEAK20_TARGET
                                else "B" if peak_net_value > 0
                                else "C"
                            ),
                            "forward20_path": path,
                        }
                    )
                elif valid_close_ret:
                    updates20["forward20_available_days"] = int(len(path))
                    updates20["forward20_path"] = path

            if any(item.get(k) != v for k, v in updates20.items()):
                changed = True
            item.update(updates20)

    if changed:
        payload["performance_updated_at"] = now_cn().isoformat()
    return changed


def enrich_payload_display_prices(payload: dict[str, Any]) -> bool:
    """Backfill public JSON display prices with raw/unadjusted market prices."""
    date_str = payload.get("date")
    if not date_str:
        return False
    raw = load_raw_close_for_date(date_str)
    if raw.empty:
        return False
    raw_map = raw.set_index("code").to_dict(orient="index")
    changed = False
    for items in payload_item_groups(payload):
        for item in items:
            code = str(item.get("code", "")).zfill(6)
            info = raw_map.get(code)
            if not info:
                continue
            raw_close = finite_float(info.get("raw_close"))
            if raw_close is None or raw_close <= 0:
                continue
            if "adjusted_close" not in item:
                old_close = finite_float(item.get("close"))
                if old_close is not None:
                    item["adjusted_close"] = round(old_close, 4)
            for c in ["raw_open", "raw_high", "raw_low"]:
                v = finite_float(info.get(c))
                if v is not None:
                    new_v = round(v, 4)
                    if item.get(c) != new_v:
                        changed = True
                    item[c] = new_v
            new_close = round(raw_close, 4)
            if item.get("close") != new_close or item.get("price_type") != "raw_unadjusted":
                changed = True
            item["raw_close"] = new_close
            item["display_close"] = new_close
            item["close"] = new_close
            item["price_type"] = "raw_unadjusted"
    if changed:
        payload["price_updated_at"] = now_cn().isoformat()
        payload["price_note"] = "网页展示的 close/display_close/raw_close 均优先使用 AkShare 原始未复权行情价，与普通行情软件看到的价格一致；模型内部特征仍使用连续复权价。"
    return changed


def refresh_history_forward_returns(rolling: pd.DataFrame) -> list[str]:
    """Backfill next-day returns for existing history JSON files."""
    updated: list[str] = []
    for path in sorted(HISTORY_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] skip malformed history {path.name}: {exc}")
            continue
        changed = enrich_payload_forward_returns(payload, rolling)
        changed = enrich_payload_display_prices(payload) or changed
        if changed:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            updated.append(payload.get("date") or path.stem)

            latest_path = PUBLIC_DIR / "latest.json"
            if latest_path.exists():
                try:
                    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
                except Exception:
                    latest_payload = {}
                if latest_payload.get("date") == payload.get("date"):
                    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if updated:
        print(f"[INFO] forward returns updated for: {', '.join(updated)}")
        write_history_index()
    return updated


def payload_strategy_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return strategy-like groups for both new multi-strategy and old legacy JSON."""
    strategies = payload.get("strategies")
    if isinstance(strategies, list) and strategies:
        return [s for s in strategies if isinstance(s, dict)]
    return [
        {
            "id": payload.get("primary_strategy_id") or "legacy",
            "name": payload.get("strategy") or "主策略",
            "short_name": "主策略",
            "items": payload.get("items") or [],
        }
    ]


def one_day_stats_for_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    dates: set[str] = set()
    for item in items:
        value = finite_float(item.get("next_1d_return"))
        if value is None:
            continue
        values.append(value)
        if item.get("next_1d_date"):
            dates.add(str(item["next_1d_date"]))
    if not values:
        return {"count": 0, "avg_return": None, "win_rate": None, "next_dates": sorted(dates)}
    arr = np.array(values, dtype=float)
    return {
        "count": int(len(values)),
        "avg_return": round(float(arr.mean()), 6),
        "win_rate": round(float((arr > 0).mean()), 6),
        "best_return": round(float(arr.max()), 6),
        "worst_return": round(float(arr.min()), 6),
        "next_dates": sorted(dates),
    }


def forward20_stats_for_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [x for x in items if x.get("forward20_complete") and finite_float(x.get("forward20_peak_net_return")) is not None]
    if not rows:
        return {"count": 0, "hit_rate": None, "avg_peak_net_return": None, "avg_end_net_return": None, "end_dates": []}
    peaks = np.array([finite_float(x.get("forward20_peak_net_return")) for x in rows], dtype=float)
    ends = np.array([finite_float(x.get("forward20_end_net_return")) for x in rows if finite_float(x.get("forward20_end_net_return")) is not None], dtype=float)
    best = max(rows, key=lambda x: finite_float(x.get("forward20_peak_net_return")) or -999.0)
    worst = min(rows, key=lambda x: finite_float(x.get("forward20_peak_net_return")) or 999.0)
    return {
        "count": int(len(rows)),
        "hit_rate": round(float((peaks >= PEAK20_TARGET).mean()), 6),
        "avg_peak_net_return": round(float(peaks.mean()), 6),
        "avg_end_net_return": round(float(ends.mean()), 6) if len(ends) else None,
        "best_peak_net_return": round(float(finite_float(best.get("forward20_peak_net_return")) or 0.0), 6),
        "best_code": best.get("code"),
        "best_name": best.get("name"),
        "worst_peak_net_return": round(float(finite_float(worst.get("forward20_peak_net_return")) or 0.0), 6),
        "worst_code": worst.get("code"),
        "worst_name": worst.get("name"),
        "end_dates": sorted({str(x.get("forward20_end_date")) for x in rows if x.get("forward20_end_date")}),
    }


def write_history_index() -> dict[str, Any]:
    """Write a static index so the website can offer a date selector."""
    entries: list[dict[str, Any]] = []
    latest_date: str | None = None
    latest_path = PUBLIC_DIR / "latest.json"
    if latest_path.exists():
        try:
            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_date = latest_payload.get("date")
        except Exception:
            latest_date = None

    for path in sorted(HISTORY_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] skip malformed history for index {path.name}: {exc}")
            continue
        date_str = payload.get("date") or path.stem
        groups = payload_strategy_groups(payload)
        all_items: list[dict[str, Any]] = []
        strategy_summaries = []
        for group in groups:
            items = group.get("items") or []
            if isinstance(items, list):
                all_items.extend([x for x in items if isinstance(x, dict)])
            stats = one_day_stats_for_items(items if isinstance(items, list) else [])
            strategy_summaries.append(
                {
                    "id": group.get("id") or "legacy",
                    "name": group.get("name") or group.get("short_name") or "主策略",
                    "candidate_count": len(items) if isinstance(items, list) else 0,
                    "one_day": stats,
                    "forward20": forward20_stats_for_items(items if isinstance(items, list) else []),
                }
            )
        entries.append(
            {
                "date": date_str,
                "is_latest": bool(latest_date and date_str == latest_date),
                "strategy_count": len(groups),
                "candidate_count": len(all_items),
                "generated_at": payload.get("generated_at"),
                "one_day": one_day_stats_for_items(all_items),
                "forward20": forward20_stats_for_items(all_items),
                "strategies": strategy_summaries,
            }
        )

    entries = sorted(entries, key=lambda x: str(x["date"]), reverse=True)
    out = {
        "generated_at": now_cn().isoformat(),
        "latest_date": latest_date or (entries[0]["date"] if entries else None),
        "dates": entries,
    }
    (PUBLIC_DIR / "history_index.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def finite_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    return f if np.isfinite(f) else None


def round_or_none(value: Any, digits: int = 6) -> float | None:
    f = finite_float(value)
    return round(f, digits) if f is not None else None


def row_price_fields(r: Any) -> dict[str, Any]:
    raw_close = finite_float(getattr(r, "raw_close", np.nan))
    raw_open = finite_float(getattr(r, "raw_open", np.nan))
    raw_high = finite_float(getattr(r, "raw_high", np.nan))
    raw_low = finite_float(getattr(r, "raw_low", np.nan))
    adjusted_close = finite_float(getattr(r, "close", np.nan))
    display_close = raw_close if raw_close is not None and raw_close > 0 else adjusted_close
    price_type = "raw_unadjusted" if raw_close is not None and raw_close > 0 else "adjusted_fallback"
    return {
        # `close` is intentionally the market-software display price. The adjusted
        # continuous price remains available for audits as `adjusted_close`.
        "close": round(display_close, 4) if display_close is not None else None,
        "display_close": round(display_close, 4) if display_close is not None else None,
        "price_type": price_type,
        "raw_open": round(raw_open, 4) if raw_open is not None else None,
        "raw_high": round(raw_high, 4) if raw_high is not None else None,
        "raw_low": round(raw_low, 4) if raw_low is not None else None,
        "raw_close": round(raw_close, 4) if raw_close is not None else None,
        "adjusted_close": round(adjusted_close, 4) if adjusted_close is not None else None,
    }


def build_strategy_items(scored: pd.DataFrame, strategy: dict[str, Any]) -> list[dict[str, Any]]:
    score_col = strategy["score_col"]
    pred_win_col = strategy.get("pred_win_col")
    pred_ret_col = strategy.get("pred_ret_col")
    metric1_col = strategy.get("metric1_col")
    metric2_col = strategy.get("metric2_col")
    top_n = int(strategy.get("top_n") or TOP_N)
    threshold = strategy.get("threshold")

    df = scored.dropna(subset=[score_col]).copy()
    if strategy.get("custom_filter") == "v12_close_only":
        df = df[df["v12_close_only_pass"]].copy()
    if threshold is not None:
        df = df[df[score_col] >= float(threshold)].copy()
    df = df.sort_values(score_col, ascending=False).head(top_n)

    items: list[dict[str, Any]] = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        pred_win = getattr(r, pred_win_col) if pred_win_col else np.nan
        pred_ret = getattr(r, pred_ret_col) if pred_ret_col else np.nan
        metric1 = getattr(r, metric1_col) if metric1_col else pred_win
        metric2 = getattr(r, metric2_col) if metric2_col else pred_ret
        item = {
            "rank": i,
            "code": str(r.code).zfill(6),
            "name": str(getattr(r, "name", "") or ""),
            "score": round(float(getattr(r, score_col)), 6),
            "pred_win": round_or_none(pred_win, 6),
            "pred_ret": round_or_none(pred_ret, 6),
            "metric1": round_or_none(metric1, 6),
            "metric2": round_or_none(metric2, 6),
            "amount": round(float(r.amount), 2),
            "turnover": round(float(r.turnover), 6),
            "daily_return": round(float(r.ret_1), 6) if pd.notna(r.ret_1) else None,
            "next_1d_date": None,
            "next_1d_return": None,
        }
        if strategy.get("custom_filter") == "v12_close_only":
            item.update(
                {
                    "v12_rank_v03a": int(getattr(r, "v12_rank_v03a")),
                    "v12_score": round(float(getattr(r, "v12_score")), 6),
                    "daily_range": round(float(getattr(r, "daily_range")), 6),
                    "v12_rule": "close_only: rank_v03a>3; exclude 16<=rank<=30 & daily_range>8.5%; no T+1 open-gap trigger",
                }
            )
        if strategy.get("model_family") == "qlib_peak20":
            item.update(
                {
                    "peak20_pred_hit": round_or_none(getattr(r, "peak20_pred_hit", None), 6),
                    "peak20_pred_peak": round_or_none(getattr(r, "peak20_pred_peak", None), 6),
                    "peak20_score": round_or_none(getattr(r, "peak20_score", None), 6),
                    "suggested_exit": "TP10或Trail8：收盘浮盈≥10%次日开盘止盈；或曾浮盈≥8%且从高点回撤≥3个百分点次日开盘卖出。",
                }
            )
        item.update(row_price_fields(r))
        items.append(item)
    return items


def build_strategy_payloads(scored: pd.DataFrame) -> list[dict[str, Any]]:
    strategies = []
    for strategy in STRATEGY_DEFS:
        items = build_strategy_items(scored, strategy)
        strategies.append(
            {
                "id": strategy["id"],
                "name": strategy["name"],
                "short_name": strategy["short_name"],
                "profile": strategy["profile"],
                "model_family": strategy["model_family"],
                "score_column": strategy["score_col"],
                "score_threshold": strategy.get("threshold"),
                "metric1_label": strategy.get("metric1_label", "胜率预测"),
                "metric1_format": strategy.get("metric1_format", "pct"),
                "metric2_label": strategy.get("metric2_label", "收益预测"),
                "metric2_format": strategy.get("metric2_format", "pct"),
                "top_n": strategy.get("top_n", TOP_N),
                "trade_rule": strategy["trade_rule"],
                "selection_note": strategy["selection_note"],
                "backtest": strategy["backtest"],
                "reports": strategy["reports"],
                "items": items,
            }
        )
    return strategies


def write_public(
    scored: pd.DataFrame,
    source: str,
    fetched: bool,
    rolling: pd.DataFrame | None = None,
    update_latest: bool = True,
) -> dict[str, Any]:
    latest_date = pd.Timestamp(scored["date"].iloc[0]).strftime("%Y-%m-%d")
    strategies = build_strategy_payloads(scored)
    primary = strategies[0]
    items = primary["items"]
    payload = {
        "strategy": primary["name"],
        "date": latest_date,
        "generated_at": now_cn().isoformat(),
        "source": source,
        "fetched_today": fetched,
        "alpha": ALPHA,
        "top_n": TOP_N,
        "universe": "A股主板 00/60，剔除 ST/退，停牌/零成交过滤",
        "trade_rule": primary["trade_rule"],
        "note": "收盘后无法提前知道次日开盘是否涨停，因此这里展示的是候选列表，不是最终成交确认。",
        "price_note": "网页展示的 close/display_close/raw_close 均优先使用 AkShare 原始未复权行情价，与普通行情软件看到的价格一致；模型内部特征仍使用连续复权价。",
        "primary_strategy_id": primary["id"],
        "items": items,
        "strategies": strategies,
    }
    if rolling is not None:
        enrich_payload_forward_returns(payload, rolling)
    enrich_payload_display_prices(payload)
    if update_latest:
        (PUBLIC_DIR / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(f"[INFO] backfill mode: wrote history for {latest_date}, preserved existing public/latest.json")
    (HISTORY_DIR / f"{latest_date}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_history_index()
    return payload


def email_html(payload: dict[str, Any]) -> str:
    def pct(v: Any) -> str:
        f = finite_float(v)
        return "" if f is None else f"{f:.2%}"

    def fmt_metric(v: Any, kind: str | None) -> str:
        f = finite_float(v)
        if f is None:
            return ""
        if kind == "pct":
            return f"{f:.2%}"
        if kind == "int":
            return f"{int(round(f))}"
        if kind == "price":
            return f"{f:.2f}"
        return f"{f:.6f}"

    sections = []
    for strategy in payload.get("strategies", []) or [{"name": payload["strategy"], "items": payload["items"]}]:
        metric1_label = strategy.get("metric1_label", "胜率分")
        metric2_label = strategy.get("metric2_label", "收益预测")
        metric1_format = strategy.get("metric1_format", "pct")
        metric2_format = strategy.get("metric2_format", "pct")
        rows = "".join(
            f"<tr><td>{x['rank']}</td><td>{x['code']}</td><td>{x['name']}</td><td>{x['score']:.6f}</td><td>{fmt_metric(x.get('metric1', x.get('pred_win')), metric1_format)}</td><td>{fmt_metric(x.get('metric2', x.get('pred_ret')), metric2_format)}</td><td>{x.get('close') or ''}</td><td>{pct(x.get('daily_return'))}</td><td>{pct(x.get('next_1d_return')) or '待更新'}</td><td>{pct(x.get('forward20_peak_net_return')) or '待完整'}</td><td>{pct(x.get('forward20_end_net_return')) or '待完整'}</td><td>{'命中' if x.get('forward20_hit8') else ('未命中' if x.get('forward20_complete') else '待完整')}</td></tr>"
            for x in strategy.get("items", [])
        )
        if not rows:
            rows = "<tr><td colspan='12' style='color:#64748b'>今日无股票达到该策略阈值</td></tr>"
        sections.append(
            f"""
            <h3>{strategy['name']}</h3>
            <table cellpadding='8' cellspacing='0' border='0' style='border-collapse:collapse;width:100%;font-size:14px;margin-bottom:18px'>
              <thead><tr style='background:#f1f5f9'><th>排名</th><th>代码</th><th>名称</th><th>分数</th><th>{metric1_label}</th><th>{metric2_label}</th><th>行情价</th><th>当日涨跌</th><th>买入后1日</th><th>20日峰值</th><th>20日期末</th><th>20日成绩</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """
        )
    return f"""
    <div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;color:#111827'>
      <h2>{payload['strategy']} - {payload['date']}</h2>
      <p>{payload['trade_rule']}</p>
      <p style='color:#64748b'>{payload['note']} 买入后1日按T+1开盘买入、T+1收盘计价；若T+1开盘涨停则实盘无法买入，需要顺位递补。</p>
      <p style='color:#64748b'>{payload.get('price_note', '')}</p>
      {''.join(sections)}
      <p><a href='{SITE_URL}'>打开网站查看</a></p>
    </div>
    """


def send_email(payload: dict[str, Any]) -> bool:
    api_key = os.getenv("RESEND_API_KEY")
    to_email = os.getenv("PICK_NOTIFY_EMAIL")
    if not api_key or not to_email:
        print("[INFO] email skipped: RESEND_API_KEY or PICK_NOTIFY_EMAIL missing")
        return False
    from_email = os.getenv("RESEND_FROM_EMAIL", "noreply@292828.xyz")
    from_name = os.getenv("RESEND_FROM_NAME", "A股每日选股")
    api_key = api_key.strip().strip("\"").strip("\'")
    if not api_key.startswith("re_") or not api_key.isascii():
        print("[ERROR] RESEND_API_KEY format looks invalid; please set a real Resend API key in GitHub Secrets")
        return False
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "from": f"{from_name} <{from_email}>",
            "to": [to_email],
            "subject": f"{payload['date']} A股每日Top10选股",
            "html": email_html(payload),
        },
        timeout=30,
    )
    if resp.status_code >= 300:
        print(f"[ERROR] resend failed: {resp.status_code} {resp.text[:300]}")
        return False
    print("[INFO] email sent")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-akshare", action="store_true", help="fetch today's close data from AkShare")
    parser.add_argument("--no-fetch", action="store_true", help="score latest cached date only")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--no-email", action="store_true")
    parser.add_argument("--date", default=os.getenv("PICK_DATE") or today_str_cn())
    parser.add_argument("--force", action="store_true", help="run even if date is not Chinese trade day")
    args = parser.parse_args()

    ensure_dirs()
    rolling = load_rolling()
    fetched = False
    names = pd.DataFrame(columns=["code", "name"])

    if args.fetch_akshare and not args.no_fetch:
        if not args.force and not is_cn_trade_day(args.date):
            print(f"[INFO] {args.date} is not CN trade day, skip")
            return
        today, source = fetch_akshare_for_date(args.date, rolling)
        names = today[["code", "name"]].copy()
        rolling = append_today(rolling, today)
        fetched = True
    else:
        source = "cached rolling_ohlcv.parquet"

    score_date = args.date if fetched else None
    update_latest = True
    if score_date is not None:
        target_ts = pd.Timestamp(score_date).normalize()
        max_ts = pd.Timestamp(rolling["date"].max()).normalize()
        update_latest = target_ts >= max_ts
        if not update_latest:
            print(
                f"[INFO] backfill target {target_ts.strftime('%Y-%m-%d')} is older than "
                f"rolling latest {max_ts.strftime('%Y-%m-%d')}; public/latest.json will be preserved"
            )

    refresh_history_forward_returns(rolling)
    panel = compute_features(rolling)
    scored = score_latest(panel, names=names, target_date=score_date)
    payload = write_public(scored, source=source, fetched=fetched, rolling=rolling, update_latest=update_latest)
    print(json.dumps({"date": payload["date"], "top1": payload["items"][0], "count": len(scored)}, ensure_ascii=False, indent=2))

    if args.send_email and not args.no_email:
        ok = send_email(payload)
        if not ok:
            print("[WARN] email was not sent, but pick result was generated successfully")


if __name__ == "__main__":
    main()
