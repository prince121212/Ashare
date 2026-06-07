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
SITE_URL = os.getenv("SITE_URL", "https://a.292828.xyz")
AKSHARE_HIST_WORKERS = int(os.getenv("AKSHARE_HIST_WORKERS", "6"))
AKSHARE_HIST_RETRIES = int(os.getenv("AKSHARE_HIST_RETRIES", "3"))

FEATURE_COLS = json.loads((MODELS_DIR / "feature_cols.json").read_text(encoding="utf-8"))


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
    out = df[["code", "name", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "raw_open", "raw_close", "raw_prev_close", "scale"]].copy()
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
            "raw_open", "raw_close", "raw_prev_date", "raw_prev_close", "scale",
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


def score_latest(panel: pd.DataFrame, names: pd.DataFrame | None = None) -> pd.DataFrame:
    latest_date = panel["date"].max()
    latest = panel[panel["date"].eq(latest_date)].dropna(subset=FEATURE_COLS).copy()
    latest = latest[latest["code"].str.startswith(("00", "60"))]
    latest = latest[(latest["open"] > 0) & (latest["close"] > 0) & (latest["amount"] > 0)]
    win_model = lgb.Booster(model_file=str(MODELS_DIR / "win_classifier.txt"))
    ret_model = lgb.Booster(model_file=str(MODELS_DIR / "return_regressor.txt"))
    x = latest[FEATURE_COLS]
    latest["pred_win"] = win_model.predict(x)
    latest["pred_ret"] = ret_model.predict(x)
    latest["score"] = latest["pred_win"] + ALPHA * latest["pred_ret"].clip(-0.05, 0.12)
    latest["rank"] = latest["score"].rank(method="first", ascending=False).astype(int)
    fallback_names = load_universe_names()
    if names is not None and not names.empty:
        merged_names = pd.concat([fallback_names, names[["code", "name"]]], ignore_index=True).drop_duplicates("code", keep="last")
    else:
        merged_names = fallback_names
    if not merged_names.empty:
        latest = latest.merge(merged_names, on="code", how="left")
    else:
        latest["name"] = ""
    return latest.sort_values("score", ascending=False).reset_index(drop=True)


def enrich_payload_forward_returns(payload: dict[str, Any], rolling: pd.DataFrame) -> bool:
    """Add next-trading-day close-to-close returns to a public payload when available."""
    if not payload.get("date") or not payload.get("items"):
        return False
    signal_date = pd.Timestamp(payload["date"]).normalize()
    dates = sorted(pd.to_datetime(rolling["date"]).dt.normalize().unique())
    next_dates = [pd.Timestamp(d).normalize() for d in dates if pd.Timestamp(d).normalize() > signal_date]
    if not next_dates:
        for item in payload["items"]:
            item.setdefault("next_1d_date", None)
            item.setdefault("next_1d_return", None)
        return False

    next_date = next_dates[0]
    signal_close = rolling[rolling["date"].eq(signal_date)].set_index("code")["close"]
    next_close = rolling[rolling["date"].eq(next_date)].set_index("code")["close"]
    changed = False
    for item in payload["items"]:
        code = str(item.get("code", "")).zfill(6)
        base = signal_close.get(code, item.get("close"))
        nxt = next_close.get(code, np.nan)
        if pd.notna(base) and pd.notna(nxt) and float(base) > 0:
            value = round(float(nxt) / float(base) - 1.0, 6)
            if item.get("next_1d_return") != value or item.get("next_1d_date") != next_date.strftime("%Y-%m-%d"):
                changed = True
            item["next_1d_date"] = next_date.strftime("%Y-%m-%d")
            item["next_1d_return"] = value
        else:
            item["next_1d_date"] = next_date.strftime("%Y-%m-%d")
            item["next_1d_return"] = None
    if changed:
        payload["performance_updated_at"] = now_cn().isoformat()
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
    return updated


def write_public(scored: pd.DataFrame, source: str, fetched: bool, rolling: pd.DataFrame | None = None) -> dict[str, Any]:
    latest_date = pd.Timestamp(scored["date"].iloc[0]).strftime("%Y-%m-%d")
    top = scored.head(TOP_N).copy()
    items = []
    for i, r in enumerate(top.itertuples(index=False), start=1):
        items.append(
            {
                "rank": i,
                "code": str(r.code).zfill(6),
                "name": str(getattr(r, "name", "") or ""),
                "score": round(float(r.score), 6),
                "pred_win": round(float(r.pred_win), 6),
                "pred_ret": round(float(r.pred_ret), 6),
                "close": round(float(r.close), 4),
                "amount": round(float(r.amount), 2),
                "turnover": round(float(r.turnover), 6),
                "daily_return": round(float(r.ret_1), 6) if pd.notna(r.ret_1) else None,
                "next_1d_date": None,
                "next_1d_return": None,
            }
        )
    payload = {
        "strategy": STRATEGY_NAME,
        "date": latest_date,
        "generated_at": now_cn().isoformat(),
        "source": source,
        "fetched_today": fetched,
        "alpha": ALPHA,
        "top_n": TOP_N,
        "universe": "A股主板 00/60，剔除 ST/退，停牌/零成交过滤",
        "trade_rule": "T日收盘打分；次日开盘按Top10顺序检查，若涨停无法买入则顺位递补；目标T+2收盘卖出，跌停顺延。",
        "note": "收盘后无法提前知道次日开盘是否涨停，因此这里展示的是候选Top10，不是最终成交确认。",
        "items": items,
    }
    if rolling is not None:
        enrich_payload_forward_returns(payload, rolling)
    (PUBLIC_DIR / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (HISTORY_DIR / f"{latest_date}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def email_html(payload: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{x['rank']}</td><td>{x['code']}</td><td>{x['name']}</td><td>{x['score']:.6f}</td><td>{x['pred_win']:.4f}</td><td>{x['pred_ret']:.2%}</td><td>{x['daily_return'] if x['daily_return'] is not None else ''}</td><td>{x.get('next_1d_return') if x.get('next_1d_return') is not None else '待更新'}</td></tr>"
        for x in payload["items"]
    )
    return f"""
    <div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;color:#111827'>
      <h2>{payload['strategy']} - {payload['date']}</h2>
      <p>{payload['trade_rule']}</p>
      <p style='color:#64748b'>{payload['note']}</p>
      <table cellpadding='8' cellspacing='0' border='0' style='border-collapse:collapse;width:100%;font-size:14px'>
        <thead><tr style='background:#f1f5f9'><th>排名</th><th>代码</th><th>名称</th><th>综合分</th><th>胜率分</th><th>收益预测</th><th>当日涨跌</th><th>1日涨跌</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
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

    refresh_history_forward_returns(rolling)
    panel = compute_features(rolling)
    scored = score_latest(panel, names=names)
    payload = write_public(scored, source=source, fetched=fetched, rolling=rolling)
    print(json.dumps({"date": payload["date"], "top1": payload["items"][0], "count": len(scored)}, ensure_ascii=False, indent=2))

    if args.send_email and not args.no_email:
        ok = send_email(payload)
        if not ok:
            print("[WARN] email was not sent, but pick result was generated successfully")


if __name__ == "__main__":
    main()
