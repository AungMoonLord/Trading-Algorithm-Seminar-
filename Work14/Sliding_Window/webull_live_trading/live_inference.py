"""
live_inference.py
==================
แกนตัดสินใจของวันนี้: ใช้ราคา+feature ล่าสุด, สถานะพอร์ตจริงจาก Webull,
และ state ของ risk engine ที่ persist ไว้ -> ได้ target weights และ orders
ที่ต้องส่งเพื่อ rebalance

หมายเหตุจังหวะเวลา: env ต้นฉบับตัดสินใจโดยใช้ข้อมูลถึง close ของวัน t แล้ว
"เข้าซื้อที่ open ของวัน t+1" ดังนั้น flow ที่ถูกต้องคือ:
    รันสคริปต์นี้ "หลังตลาดปิด" ของวันนี้ -> ได้ orders -> ส่งเป็น MARKET-ON-OPEN
    (หรือ LIMIT ใกล้ open) สำหรับ "พรุ่งนี้เช้า"
ถ้าส่งเป็น MARKET order ทันทีตอนกลางคืน ให้ตั้งเวลาส่งช่วงก่อนตลาดเปิดของ
วันถัดไปแทน อย่าส่งดึกแล้วปล่อยเป็น market order ค้างคืนข้าม gap เสี่ยงเกินจำเป็น
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import common as C
import state_manager as SM


@dataclass
class RebalanceOrder:
    ticker: str
    side: str          # "BUY" | "SELL"
    target_weight: float
    current_weight: float
    delta_notional: float   # +ซื้อ / -ขาย (USD โดยประมาณ ใช้ราคาล่าสุดคำนวณ qty เอง)
    approx_qty: int


@dataclass
class DecisionResult:
    target_weights: Dict[str, float]   # รวม "CASH"
    current_weights: Dict[str, float]
    turnover: float
    scalar: float
    realized_vol: float
    orders: List[RebalanceOrder]
    raw_action: np.ndarray


def load_model(algo: str, model_path: Path):
    algo = algo.lower()
    if algo == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo == "a2c":
        from stable_baselines3 import A2C as Algo
    elif algo == "sac":
        from stable_baselines3 import SAC as Algo
    elif algo == "td3":
        from stable_baselines3 import TD3 as Algo
    else:
        raise ValueError(f"ไม่รู้จัก algo: {algo} (ต้องเป็น ppo/a2c/sac/td3)")
    return Algo.load(str(model_path))


def get_latest_scaled_row(clean_full_history: pd.DataFrame, scaling_stats: dict) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """เอาข้อมูลวันล่าสุดที่มีครบทุก ticker แล้ว apply scaling (เหมือน apply_scaling บน test_slice)"""
    latest_date = clean_full_history["date"].max()
    today_raw = clean_full_history[clean_full_history["date"] == latest_date].copy()
    today_scaled = C.apply_scaling(today_raw, scaling_stats)
    return today_scaled, latest_date


def decide(
    model_name: str,           # ชื่อไฟล์ state เช่น "ppo"
    algo: str,                 # "ppo" | "a2c" | "sac" | "td3"
    model_path: Path,
    state_dir: Path,           # schemes/<scheme>/state/ — แยก state ตาม scheme
    clean_full_history: pd.DataFrame,   # ผลจาก fetch_and_stitch_universe -> compute_institutional_features -> clean_and_align_dataset (ข้อมูลล่าสุดถึงวันนี้)
    scaling_stats: dict,
    current_equity: float,             # net liquidation value จากบัญชี Webull วันนี้
    current_position_values: Dict[str, float],  # market value ต่อ ticker จากบัญชี Webull วันนี้ (ไม่รวมเงินสด)
    latest_prices: Dict[str, float],   # ราคาล่าสุดต่อ ticker (ใช้ประมาณ qty)
) -> DecisionResult:
    tickers = C.CFG.core_tickers
    today_scaled, latest_date = get_latest_scaled_row(clean_full_history, scaling_stats)
    today_iso = latest_date.strftime("%Y-%m-%d")

    # ---- โหลด / init state ของ risk engine ----
    state = SM.load_state(state_dir, model_name)
    if state is None:
        state = SM.init_state(state_dir, model_name, current_equity, C.CFG.target_annual_vol,
                               C.CFG.vol_ewma_lambda, today_iso)
        print(f"[{model_name}] ยังไม่มี state -> cold start ที่ equity={current_equity:,.2f}")

    risk_engine = C.VolatilityTargetingEngine.from_state({
        "variance": state.variance, "last_scalar": state.last_scalar,
        "target_daily_vol": state.target_daily_vol, "decay": state.decay,
    })

    portfolio_step_return = (current_equity / state.last_portfolio_value - 1.0) if state.last_portfolio_value else 0.0
    peak_value = max(state.peak_value or current_equity, current_equity)
    current_drawdown = max(0.0, 1.0 - (current_equity / peak_value))

    # ---- current weights จากบัญชีจริง (obs_weights ของวันนี้ ก่อน rebalance) ----
    cash_value = current_equity - sum(current_position_values.get(t, 0.0) for t in tickers)
    current_weights = np.array(
        [cash_value] + [current_position_values.get(t, 0.0) for t in tickers]
    ) / current_equity

    # ---- observation + model ----
    obs = C.build_today_observation(
        today_scaled, tickers, current_weights,
        scalar=risk_engine.last_scalar, realized_vol=np.sqrt(risk_engine.variance),
        target_daily_vol=risk_engine.target_daily_vol,
    )
    model = load_model(algo, model_path)
    action, _ = model.predict(obs, deterministic=True)

    target_weights, scalar, realized_vol = C.action_to_target_weights(
        action, risk_engine, portfolio_step_return, current_drawdown
    )
    final_weights, turnover = C.apply_no_trade_band(target_weights, current_weights)

    # ---- persist state ใหม่ ----
    new_state = SM.LiveState(
        last_date=today_iso, last_portfolio_value=current_equity, peak_value=peak_value,
        variance=risk_engine.variance, last_scalar=risk_engine.last_scalar,
        target_daily_vol=risk_engine.target_daily_vol, decay=risk_engine.decay,
    )
    SM.save_state(state_dir, model_name, new_state)

    # ---- แปลงเป็น orders ----
    orders: List[RebalanceOrder] = []
    if turnover > 0:
        for i, tic in enumerate(tickers, start=1):
            delta_w = final_weights[i] - current_weights[i]
            delta_notional = delta_w * current_equity
            if abs(delta_notional) < 1.0:
                continue
            price = latest_prices.get(tic)
            qty = int(abs(delta_notional) // price) if price else 0
            if qty <= 0:
                continue
            orders.append(RebalanceOrder(
                ticker=tic, side="BUY" if delta_notional > 0 else "SELL",
                target_weight=float(final_weights[i]), current_weight=float(current_weights[i]),
                delta_notional=float(delta_notional), approx_qty=qty,
            ))

    return DecisionResult(
        target_weights={"CASH": float(final_weights[0]), **{t: float(final_weights[i+1]) for i, t in enumerate(tickers)}},
        current_weights={"CASH": float(current_weights[0]), **{t: float(current_weights[i+1]) for i, t in enumerate(tickers)}},
        turnover=turnover, scalar=scalar, realized_vol=realized_vol,
        orders=orders, raw_action=action,
    )
