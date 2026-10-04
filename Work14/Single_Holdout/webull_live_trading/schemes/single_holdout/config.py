"""
schemes/single_holdout/config.py
==================================
✅ ยืนยันแล้วจาก Single_Holdout_PPO_synced.ipynb ที่ส่งมา

Single Holdout = train/val/test แบ่งครั้งเดียว ไม่ทำ walk-forward หลาย fold
(WALK_FORWARD_FOLDS ใน notebook มีแค่ 1 tuple เดียว) feature list, risk engine,
และค่า config อื่นๆ (target_annual_vol, transaction_cost_pct ฯลฯ) เหมือนกับ
Sliding Window / Expanding Window ทุกตัว — common.py ใช้ร่วมกันได้โดยไม่ต้องแก้
"""
from pathlib import Path

SCHEME_NAME = "single_holdout"
SCHEME_DIR = Path(__file__).parent

# ตรงกับ WALK_FORWARD_FOLDS ใน notebook เป๊ะ (มี fold เดียว ไม่ใช่ walk-forward)
TRAIN_START = "1999-04-01"
TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-07-29"

# ไม่มีหลาย fold แบบ walk-forward (มี split เดียว) แต่ตั้งเป็น "fold_1" เพื่อให้
# โครงสร้างโฟลเดอร์ models/<algo>/fold_<N>/best_model.zip ใช้ร่วมกับ scheme อื่นได้
ACTIVE_FOLD_ID = 1

MODELS_DIR = SCHEME_DIR / "models"
STATE_DIR = SCHEME_DIR / "state"
SCALING_STATS_PATH = STATE_DIR / "scaling_stats.json"

NOTES = (
    f"Single Holdout: Train [{TRAIN_START}:{TRAIN_END}] -> "
    f"Val [{VAL_START}:{VAL_END}] -> Test [{TEST_START}:{TEST_END}]"
)