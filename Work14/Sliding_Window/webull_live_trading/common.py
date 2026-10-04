"""
common.py
=========
โค้ดชุดนี้ "คัดลอกมาจากของจริง" ใน 4 notebook (PPO/A2C/SAC/TD3) เฉพาะส่วนที่
เหมือนกันทุกตัว (ยืนยันแล้วว่า ALL_FEATURES, core_tickers, WALK_FORWARD_FOLDS,
obs_dim formula, action_space เหมือนกัน 100% ทั้ง 4 ไฟล์)

เป้าหมาย: ให้ observation ที่ป้อนเข้าโมเดลตอน live เหมือนกับตอนเทรนทุกประการ
ไม่งั้นโมเดลจะเห็นข้อมูลผิดสเกล -> action ผิดเพี้ยน

ต้องรันบนเครื่องที่ต่อเน็ตได้จริง (yfinance + FRED) — Claude รันให้ไม่ได้เพราะ
sandbox นี้ whitelist เฉพาะ pypi/npm/github ห้าม query1.finance.yahoo.com และ
fred ครับ โค้ดชุดนี้ผ่านการทดสอบด้วย synthetic data (เหมือน audit suite ใน
notebook ต้นฉบับ) แต่ยังไม่เคยรันกับข้อมูลจริง ให้รันเทียบผลกับ oos_account.csv
ของ fold 5 ก่อนใช้งานจริงเสมอ (ดู README ข้อ "การตรวจสอบก่อนใช้งานจริง")
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG (เฉพาะค่าที่จำเป็นต่อ inference — ตัดส่วน hyperparameter การเทรนออก)
# ==============================================================================
@dataclass(frozen=True)
class LiveConfig:
    core_tickers: Tuple[str, ...] = ("SPY", "QQQ", "DIA", "TLT", "GLD")
    initial_amount: float = 1_000_000.0          # ใช้แค่ตอน sanity-check เท่านั้น
    transaction_cost_pct: float = 0.0015
    min_turnover_threshold: float = 0.01          # no-trade band 1%
    target_annual_vol: float = 0.10
    vol_ewma_lambda: float = 0.94
    min_exposure: float = 0.20
    max_exposure: float = 1.00
    hysteresis_band: float = 0.05
    fed_funds_lag_days: int = 35
    # ต้อง fetch ข้อมูลย้อนหลังพอสำหรับ EMA200/SMA200 ก่อนวันแรกของ train window
    # ของ fold ที่ใช้จริง ไม่งั้น indicator ช่วงต้นจะเพี้ยน
    global_fetch_start: str = "1996-01-01"


CFG = LiveConfig()

# หมายเหตุ: ค่า train/val/test window ของแต่ละ scheme (Sliding Window /
# Expanding Window / Single Holdout) ย้ายไปอยู่ที่ schemes/<scheme>/config.py
# แล้ว เพราะแต่ละ scheme มีวันที่ train ต่างกัน — common.py นี้เก็บเฉพาะโค้ด
# ที่ "เหมือนกันทุก scheme ทุก algorithm" เท่านั้น

TECHNICAL_FEATURES = [
    "RSI_14", "RSI_rel", "MACD_hist_pct", "EMA_12_26_ratio",
    "EMA_50_200_ratio", "price_to_EMA200", "StochRSI_K", "mom_20d", "mom_63d"
]
CROSS_SECTIONAL_FEATURES = ["ratio_QQQ_SPY", "ratio_SPY_TLT", "mom_rank_63d"]
MACRO_FEATURES = ["vix_log", "yield_curve_slope", "gold_logret", "wti_logret", "fed_funds_rate"]
ALL_FEATURES = TECHNICAL_FEATURES + CROSS_SECTIONAL_FEATURES + MACRO_FEATURES

K = len(CFG.core_tickers)
OBS_DIM = (K * len(ALL_FEATURES)) + (K + 1) + 2  # = 88


def ensure_naive_datetime(dt_obj):
    idx = pd.DatetimeIndex(dt_obj)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _extract_price_df(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        return raw[field]
    return raw[[field]].rename(columns={field: raw.columns[0]})


# ==============================================================================
# SECTION 1: DATA ACQUISITION (เหมือนต้นฉบับ 100% — ต้องมีเน็ตจริง)
# ==============================================================================
def fetch_and_stitch_universe(tickers: Tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf
    import pandas_datareader.data as web

    print(f">> Fetching Market Data & Backcasting Inception Gaps: {start_date} -> {end_date}")
    all_tickers = list(tickers) + ["GC=F", "VUSTX", "^VIX", "^TNX", "^IRX", "CL=F"]
    raw = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)

    opens = _extract_price_df(raw, "Open")
    closes = _extract_price_df(raw, "Close")
    highs = _extract_price_df(raw, "High")
    lows = _extract_price_df(raw, "Low")
    volumes = _extract_price_df(raw, "Volume")

    opens = opens.fillna(closes).ffill()
    highs = highs.fillna(closes).ffill()
    lows = lows.fillna(closes).ffill()
    closes = closes.ffill()
    volumes = volumes.fillna(1_000_000.0)

    gld_series = closes["GLD"].dropna()
    gld_start = gld_series.index[0]
    gld_anchor = gld_series.loc[gld_start]
    gc_anchor = closes["GC=F"].asof(gld_start)
    if pd.isna(gc_anchor) or gc_anchor <= 0:
        gc_anchor = closes["GC=F"].dropna().iloc[0]
    gld_ratio = gld_anchor / gc_anchor
    pre_gld_mask = closes.index < gld_start
    days_back_gld = (pd.to_datetime(gld_start) - pd.to_datetime(closes.index[pre_gld_mask])).days
    expense_factors_gld = (1.0 - 0.0040 / 365.25) ** days_back_gld

    closes.loc[pre_gld_mask, "GLD"] = closes["GC=F"].loc[pre_gld_mask] * gld_ratio * expense_factors_gld
    opens.loc[pre_gld_mask, "GLD"] = opens["GC=F"].loc[pre_gld_mask] * gld_ratio * expense_factors_gld
    highs.loc[pre_gld_mask, "GLD"] = highs["GC=F"].loc[pre_gld_mask] * gld_ratio * expense_factors_gld
    lows.loc[pre_gld_mask, "GLD"] = lows["GC=F"].loc[pre_gld_mask] * gld_ratio * expense_factors_gld
    volumes.loc[pre_gld_mask, "GLD"] = 1_000_000.0

    tlt_series = closes["TLT"].dropna()
    tlt_start = tlt_series.index[0]
    tlt_anchor = tlt_series.loc[tlt_start]
    vustx_anchor = closes["VUSTX"].asof(tlt_start)
    if pd.isna(vustx_anchor) or vustx_anchor <= 0:
        vustx_anchor = closes["VUSTX"].dropna().iloc[0]
    tlt_ratio = tlt_anchor / vustx_anchor
    pre_tlt_mask = closes.index < tlt_start

    closes.loc[pre_tlt_mask, "TLT"] = closes["VUSTX"].loc[pre_tlt_mask] * tlt_ratio
    opens.loc[pre_tlt_mask, "TLT"] = opens["VUSTX"].loc[pre_tlt_mask] * tlt_ratio
    highs.loc[pre_tlt_mask, "TLT"] = highs["VUSTX"].loc[pre_tlt_mask] * tlt_ratio
    lows.loc[pre_tlt_mask, "TLT"] = lows["VUSTX"].loc[pre_tlt_mask] * tlt_ratio
    volumes.loc[pre_tlt_mask, "TLT"] = 1_000_000.0

    records = []
    for tic in tickers:
        df_t = pd.DataFrame({
            "date": closes.index, "tic": tic,
            "open": opens[tic].values, "high": highs[tic].values,
            "low": lows[tic].values, "close": closes[tic].values,
            "volume": volumes[tic].values,
        })
        records.append(df_t)
    panel_df = pd.concat(records, ignore_index=True)
    panel_df["date"] = ensure_naive_datetime(panel_df["date"])

    macro_df = pd.DataFrame(index=closes.index)
    macro_df["date"] = ensure_naive_datetime(closes.index)
    macro_df["vix_log"] = np.log(closes["^VIX"].clip(lower=1.0)).ffill().bfill().values
    macro_df["yield_curve_slope"] = (closes["^TNX"] - closes["^IRX"]).ffill().bfill().values
    macro_df["gold_logret"] = np.log(closes["GC=F"].clip(lower=1e-3)).diff().clip(-0.2, 0.2).fillna(0.0).values
    macro_df["wti_logret"] = np.log(closes["CL=F"].clip(lower=1e-3)).diff().clip(-0.2, 0.2).fillna(0.0).values

    try:
        fed = web.DataReader("FEDFUNDS", "fred", start_date, end_date).reset_index()
        fed.columns = ["date", "fed_raw"]
        fed["date"] = ensure_naive_datetime(fed["date"]) + pd.Timedelta(days=CFG.fed_funds_lag_days)
        macro_df = pd.merge_asof(macro_df.sort_values("date"), fed.sort_values("date"), on="date", direction="backward")
        macro_df["fed_funds_rate"] = macro_df["fed_raw"].fillna(1.0) / 100.0
    except Exception:
        macro_df["fed_funds_rate"] = 0.02

    full_df = pd.merge(panel_df, macro_df, on="date", how="inner")
    return full_df.sort_values(["date", "tic"]).reset_index(drop=True)


# ==============================================================================
# SECTION 2: FEATURE ENGINEERING (เหมือนต้นฉบับ 100%)
# ==============================================================================
def compute_institutional_features(df: pd.DataFrame, tickers: Tuple[str, ...]) -> pd.DataFrame:
    import ta

    df = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)
    processed_tics = []
    for tic, group in df.groupby("tic", sort=False):
        g = group.copy()
        close = g["close"].astype(float)

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
        g["RSI_14"] = (rsi / 100.0) - 0.5
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        ema26 = ta.trend.EMAIndicator(close, window=26).ema_indicator().replace(0, np.nan)
        g["MACD_hist_pct"] = macd.macd_diff() / ema26
        ema12 = ta.trend.EMAIndicator(close, window=12).ema_indicator()
        ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator().replace(0, np.nan)
        sma200 = close.rolling(200).mean().replace(0, np.nan)

        g["EMA_12_26_ratio"] = (ema12 - ema26) / ema26
        g["EMA_50_200_ratio"] = (ema50 - ema200) / ema200
        g["price_to_EMA200"] = (close - ema200) / ema200

        stoch = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        g["StochRSI_K"] = stoch.stochrsi_k() - 0.5

        g["mom_20d"] = close.pct_change(20).clip(-0.5, 0.5)
        g["mom_63d"] = close.pct_change(63).clip(-0.5, 0.5)

        g["downtrend_flag"] = ((close < sma200) & (g["mom_20d"] < 0)).astype(np.float32)
        processed_tics.append(g)

    df = pd.concat(processed_tics, ignore_index=True)

    pivoted_rsi = df.pivot(index="date", columns="tic", values="RSI_14")
    mean_rsi = pivoted_rsi.mean(axis=1)
    df["RSI_rel"] = df["RSI_14"] - df["date"].map(mean_rsi)

    pivoted_close = df.pivot(index="date", columns="tic", values="close")
    ratio_qqq_spy = (pivoted_close["QQQ"] / pivoted_close["SPY"]).pct_change(20)
    ratio_spy_tlt = (pivoted_close["SPY"] / pivoted_close["TLT"]).pct_change(20)
    df["ratio_QQQ_SPY"] = df["date"].map(ratio_qqq_spy).fillna(0.0)
    df["ratio_SPY_TLT"] = df["date"].map(ratio_spy_tlt).fillna(0.0)

    pivoted_mom63 = df.pivot(index="date", columns="tic", values="mom_63d")
    rank_df = (
        pivoted_mom63.rank(axis=1, pct=True)
        .reset_index()
        .melt(id_vars="date", value_name="mom_rank_63d")
    )
    df = pd.merge(df, rank_df, on=["date", "tic"], how="left")
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def clean_and_align_dataset(df: pd.DataFrame, tickers: Tuple[str, ...]) -> pd.DataFrame:
    df = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)
    for col in ALL_FEATURES + ["open", "close"]:
        df[col] = df.groupby("tic")[col].ffill()
    df = df.dropna(subset=ALL_FEATURES + ["open", "close"]).copy()

    counts = df.groupby("date")["tic"].nunique()
    valid_dates = counts[counts == len(tickers)].index
    df = df[df["date"].isin(valid_dates)].copy()

    for col in ALL_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ALL_FEATURES).copy()
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def compute_scaling_stats(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for col in ALL_FEATURES:
        series = train_df[col].dropna()
        mean = float(series.mean())
        std = float(series.std())
        stats[col] = (mean, std if std > 1e-6 else 1.0)
    return stats


def apply_scaling(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]], clip_sigma: float = 5.0) -> pd.DataFrame:
    df = df.copy()
    for col, (mean, std) in stats.items():
        if col in df.columns:
            df[col] = ((df[col] - mean) / std).clip(-clip_sigma, clip_sigma)
    return df


def save_scaling_stats(stats: Dict[str, Tuple[float, float]], path: Path) -> None:
    path.write_text(json.dumps({k: list(v) for k, v in stats.items()}, indent=2, ensure_ascii=False))


def load_scaling_stats(path: Path) -> Dict[str, Tuple[float, float]]:
    raw = json.loads(path.read_text())
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


# ==============================================================================
# SECTION 3: RISK ENGINE (เหมือนต้นฉบับ 100%) — ต้อง "stateful" ข้ามวัน
# ==============================================================================
class VolatilityTargetingEngine:
    def __init__(self, target_annual_vol: float = 0.10, ewma_lambda: float = 0.94):
        self.target_daily_vol = target_annual_vol / np.sqrt(252.0)
        self.decay = ewma_lambda
        self.variance = (self.target_daily_vol) ** 2
        self.last_scalar = 1.0

    def reset(self):
        self.variance = (self.target_daily_vol) ** 2
        self.last_scalar = 1.0

    def update_and_scale(self, portfolio_step_return: float, current_drawdown: float) -> Tuple[float, float]:
        self.variance = self.decay * self.variance + (1.0 - self.decay) * (portfolio_step_return ** 2)
        realized_vol = np.sqrt(max(self.variance, 1e-8))
        raw_scalar = float(self.target_daily_vol / realized_vol)

        if current_drawdown > 0.15:
            dd_penalty = 1.0 - min(0.50, (current_drawdown - 0.15) * 2.0)
            raw_scalar *= dd_penalty

        applied_scalar = float(np.clip(raw_scalar, CFG.min_exposure, CFG.max_exposure))

        if abs(applied_scalar - self.last_scalar) < CFG.hysteresis_band:
            applied_scalar = self.last_scalar
        else:
            self.last_scalar = applied_scalar

        return applied_scalar, realized_vol

    def to_state(self) -> dict:
        return {"variance": self.variance, "last_scalar": self.last_scalar,
                "target_daily_vol": self.target_daily_vol, "decay": self.decay}

    @classmethod
    def from_state(cls, state: dict) -> "VolatilityTargetingEngine":
        eng = cls.__new__(cls)
        eng.target_daily_vol = state["target_daily_vol"]
        eng.decay = state["decay"]
        eng.variance = state["variance"]
        eng.last_scalar = state["last_scalar"]
        return eng


def softmax(actions: np.ndarray) -> np.ndarray:
    shifted = actions - np.max(actions)
    exp_a = np.exp(shifted)
    return exp_a / np.sum(exp_a)


# ==============================================================================
# SECTION 4: OBSERVATION BUILDER สำหรับวันนี้ (แทนที่ Gym env ตอน live)
# ==============================================================================
def build_today_observation(
    features_row: pd.DataFrame,   # 1 แถวต่อ ticker (k แถว) ของ "วันล่าสุดที่มีข้อมูลครบ" หลัง apply_scaling แล้ว
    tickers: Tuple[str, ...],
    obs_weights: np.ndarray,      # [cash_w, w_1..w_k] น้ำหนักพอร์ตปัจจุบันจริงจากบัญชี ก่อน rebalance วันนี้ (ผลรวม=1)
    scalar: float,                # risk scalar ล่าสุดจาก VolatilityTargetingEngine (ค่าที่ "จะใช้" วันนี้ ไม่ใช่เมื่อวาน)
    realized_vol: float,
    target_daily_vol: float,
) -> np.ndarray:
    """
    ประกอบ observation ให้ตรงกับ _get_obs() ของ InstitutionalPortfolioEnv ทุกประการ:
        feats(k*16) + obs_weights(k+1) + [scalar, realized_vol/target_daily_vol]
    """
    assert len(obs_weights) == len(tickers) + 1
    assert np.isclose(np.sum(obs_weights), 1.0, atol=1e-4), f"obs_weights ต้องรวมเป็น 1, ได้ {np.sum(obs_weights)}"

    feat_rows = []
    for tic in tickers:
        row = features_row[features_row["tic"] == tic]
        if len(row) != 1:
            raise ValueError(f"ต้องมีข้อมูล {tic} วันล่าสุดพอดี 1 แถว, พบ {len(row)}")
        feat_rows.append(row[ALL_FEATURES].to_numpy(dtype=np.float32).reshape(-1))
    feats = np.concatenate(feat_rows)

    risk_feats = np.array([scalar, realized_vol / target_daily_vol], dtype=np.float32)
    obs = np.concatenate([feats, obs_weights.astype(np.float32), risk_feats]).astype(np.float32)

    if obs.shape[0] != OBS_DIM:
        raise ValueError(f"obs shape ผิด: ได้ {obs.shape[0]} ต้องการ {OBS_DIM}")
    return obs


def action_to_target_weights(
    action: np.ndarray, risk_engine: VolatilityTargetingEngine,
    portfolio_step_return: float, current_drawdown: float,
) -> Tuple[np.ndarray, float, float]:
    """คืนค่า (target_weights[cash,+k tickers], scalar_used, realized_vol) เหมือน step() ในต้นฉบับ"""
    raw_risky_weights = softmax(action)
    scalar, realized_vol = risk_engine.update_and_scale(portfolio_step_return, current_drawdown)
    final_risky_weights = raw_risky_weights * scalar
    cash_weight = max(0.0, 1.0 - float(np.sum(final_risky_weights)))
    target_weights = np.concatenate([[cash_weight], final_risky_weights])
    target_weights /= np.sum(target_weights)
    return target_weights, scalar, realized_vol


def apply_no_trade_band(target_weights: np.ndarray, current_weights: np.ndarray) -> Tuple[np.ndarray, float]:
    turnover_candidate = float(np.sum(np.abs(target_weights[1:] - current_weights[1:])))
    if turnover_candidate < CFG.min_turnover_threshold:
        return current_weights.copy(), 0.0
    return target_weights, turnover_candidate
