"""
schemes/expanding_window/config.py
====================================
✅ ยืนยันแล้วจาก Expand_Window_PPO.ipynb ที่ส่งมา — ตรงกับที่เคย
เดาไว้ล่วงหน้าทุกตัวเลข (WALK_FORWARD_FOLDS, target_annual_vol, transaction_cost_pct,
hysteresis_band ฯลฯ เหมือน sliding_window 100%) ต่างกันแค่ TRAIN_START ที่ตรึง
คงที่ 2000-01-01 ทุก fold ตามหลัก Expanding Window

วิธีคิด Expanding Window: train_start คงที่ แต่ train_end ขยับออกไปเรื่อยๆ ทุก
fold (ข้อมูล train ยิ่งมากขึ้นเรื่อยๆ ไม่ทิ้งข้อมูลเก่า)
"""
from pathlib import Path

SCHEME_NAME = "expanding_window"
SCHEME_DIR = Path(__file__).parent

ALL_FOLDS = [
    ("2000-01-01", "2013-12-31", "2014-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2000-01-01", "2015-12-31", "2016-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2000-01-01", "2017-12-31", "2018-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2000-01-01", "2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2000-01-01", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2026-07-29"),
]

# Fold ที่ใช้ deploy จริงตอนนี้ = fold ล่าสุด (index 5 ตรงกับ best_model.zip fold_5)
ACTIVE_FOLD_ID = 5
TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = ALL_FOLDS[ACTIVE_FOLD_ID - 1]

MODELS_DIR = SCHEME_DIR / "models"
STATE_DIR = SCHEME_DIR / "state"
SCALING_STATS_PATH = STATE_DIR / "scaling_stats.json"

NOTES = (
    f"Expanding Window fold {ACTIVE_FOLD_ID}: Train [{TRAIN_START}:{TRAIN_END}] "
    f"(train_start คงที่ ขยายออกทุก fold) -> Val [{VAL_START}:{VAL_END}] -> Test [{TEST_START}:{TEST_END}]"
)