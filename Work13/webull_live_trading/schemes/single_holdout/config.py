"""
schemes/single_holdout/config.py
==================================
⚠️ PLACEHOLDER — ยังไม่มี notebook เทรนจริงของ scheme นี้ (คุณบอกว่าจะส่งมาทีหลัง)

Single Holdout = train/val/test แบ่งครั้งเดียว ไม่ทำ walk-forward หลาย fold
ค่าด้านล่างเป็นค่าเดาไว้ก่อน (train ใช้ข้อมูลยาวที่สุดเท่าที่มี เผื่อ val ปีสุดท้าย
ก่อน test, test = ช่วงล่าสุด) **ต้องแก้ให้ตรงกับ notebook จริงก่อนใช้งาน**
"""
from pathlib import Path

SCHEME_NAME = "single_holdout"
SCHEME_DIR = Path(__file__).parent

# TODO: แทนที่ด้วยวันที่จริงจาก notebook Single Holdout ของคุณ
TRAIN_START = "2000-01-01"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-07-29"

MODELS_DIR = SCHEME_DIR / "models"
STATE_DIR = SCHEME_DIR / "state"
SCALING_STATS_PATH = STATE_DIR / "scaling_stats.json"

NOTES = (
    f"[PLACEHOLDER] Single Holdout: Train [{TRAIN_START}:{TRAIN_END}] -> "
    f"Val [{VAL_START}:{VAL_END}] -> Test [{TEST_START}:{TEST_END}] — ยังไม่ยืนยันกับ notebook จริง"
)
