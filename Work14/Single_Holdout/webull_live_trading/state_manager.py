"""
state_manager.py
=================
Risk engine (VolatilityTargetingEngine) เป็น stateful — ต้องมี variance (EWMA)
และ peak_value (สำหรับคำนวณ drawdown) ต่อเนื่องข้ามวัน ถ้ารันสคริปต์ใหม่ทุกวัน
โดยไม่ persist state จะเหมือน "รีเซ็ตความจำ" ของ risk engine ทุกวัน ทำให้ scalar
ผิดเพี้ยนจากที่โมเดลถูกเทรนมา

ไฟล์ state เก็บที่ state/live_state_<model_name>.json แยกตามโมเดล เพราะแต่ละ
โมเดล(PPO/A2C/SAC/TD3)ควรมี risk-engine state ของตัวเอง ถ้าจะรันหลายโมเดลพร้อมกัน
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LiveState:
    last_date: Optional[str] = None          # ISO date ของวันที่รันล่าสุด
    last_portfolio_value: Optional[float] = None
    peak_value: Optional[float] = None
    variance: Optional[float] = None          # risk engine EWMA variance
    last_scalar: float = 1.0
    target_daily_vol: Optional[float] = None
    decay: Optional[float] = None


def state_path(state_dir: Path, model_name: str) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"live_state_{model_name}.json"


def load_state(state_dir: Path, model_name: str) -> Optional[LiveState]:
    p = state_path(state_dir, model_name)
    if not p.exists():
        return None
    return LiveState(**json.loads(p.read_text()))


def save_state(state_dir: Path, model_name: str, state: LiveState) -> None:
    state_path(state_dir, model_name).write_text(json.dumps(asdict(state), indent=2))


def init_state(state_dir: Path, model_name: str, initial_equity: float, target_annual_vol: float,
               ewma_lambda: float, today_iso: str) -> LiveState:
    """เรียกตอนรันครั้งแรกของโมเดลนั้นๆ เท่านั้น (cold start = สมมติว่า vol อยู่ที่ target พอดี)"""
    import numpy as np
    target_daily_vol = target_annual_vol / np.sqrt(252.0)
    state = LiveState(
        last_date=today_iso,
        last_portfolio_value=initial_equity,
        peak_value=initial_equity,
        variance=target_daily_vol ** 2,
        last_scalar=1.0,
        target_daily_vol=target_daily_vol,
        decay=ewma_lambda,
    )
    save_state(state_dir, model_name, state)
    return state
