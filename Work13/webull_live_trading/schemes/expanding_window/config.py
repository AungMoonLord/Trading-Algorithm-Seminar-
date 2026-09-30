"""
schemes/expanding_window/config.py
====================================
⚠️ PLACEHOLDER — ยังไม่มี notebook เทรนจริงของ scheme นี้ (คุณบอกว่าจะส่งมาทีหลัง)

ค่าด้านล่างเป็นค่าที่ผม "เดาตามธรรมเนียม Expanding Window ทั่วไป" ไว้ก่อน โดย
ยึดหลัก: train_start ตรึงคงที่ (ไม่ตัดข้อมูลเก่าทิ้งเหมือน sliding window),
ส่วน val/test window ใช้ชุดเดียวกับ Sliding Window (5 คู่เดิม) เพื่อให้เทียบผล
กันระหว่าง 2 scheme ได้ตรงช่วงเวลา — **ต้องแก้ตัวเลขพวกนี้ให้ตรงกับ notebook
จริงที่จะส่งมาก่อนใช้งาน** (ตัวเลข val/test ไม่กระทบ scaling stats เพราะ
scaling stats คำนวณจาก train_slice เท่านั้น แต่ train_start/train_end กระทบ
โดยตรง — ผิดแล้ว scaling stats จะไม่ตรงกับที่โมเดลถูกเทรนมา)

วิธีคิด Expanding Window: train_start คงที่ แต่ train_end ขยับออกไปเรื่อยๆ ทุก
fold (ข้อมูล train ยิ่งมากขึ้นเรื่อยๆ ไม่ทิ้งข้อมูลเก่า)
"""
from pathlib import Path

SCHEME_NAME = "expanding_window"
SCHEME_DIR = Path(__file__).parent

# TODO: แทนที่ด้วย fold definitions จริงจาก notebook Expanding Window ของคุณ
ALL_FOLDS = [
    ("2000-01-01", "2013-12-31", "2014-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2000-01-01", "2015-12-31", "2016-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2000-01-01", "2017-12-31", "2018-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2000-01-01", "2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2000-01-01", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2026-07-29"),
]

ACTIVE_FOLD_ID = 5  # TODO: ยืนยันว่า fold ไหนคือ fold ล่าสุดใน notebook จริง
TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = ALL_FOLDS[ACTIVE_FOLD_ID - 1]

MODELS_DIR = SCHEME_DIR / "models"
STATE_DIR = SCHEME_DIR / "state"
SCALING_STATS_PATH = STATE_DIR / "scaling_stats.json"

NOTES = (
    f"[PLACEHOLDER] Expanding Window fold {ACTIVE_FOLD_ID}: Train [{TRAIN_START}:{TRAIN_END}] "
    f"(train_start คงที่ ขยายออกทุก fold) -> Val [{VAL_START}:{VAL_END}] -> Test [{TEST_START}:{TEST_END}] "
    f"— ยังไม่ยืนยันกับ notebook จริง"
)
