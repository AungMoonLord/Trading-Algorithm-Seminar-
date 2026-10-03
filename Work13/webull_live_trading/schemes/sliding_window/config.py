"""
schemes/sliding_window/config.py
=================================
พร้อมใช้งานจริง — ตรงกับ WALK_FORWARD_FOLDS ใน Sliding_Window_*.ipynb ทั้ง 3 ไฟล์
(PPO) ที่อัปโหลดมา ยืนยันแล้วว่าเหมือนกันทุกไฟล์

วิธีคิด Sliding Window: หน้าต่าง train มีขนาดคงที่ (14 ปี) แล้วเลื่อนไปข้างหน้า
ทุก fold (ตัดข้อมูลเก่าสุดทิ้งเมื่อเพิ่มข้อมูลใหม่เข้ามา)
"""
from pathlib import Path

SCHEME_NAME = "sliding_window"
SCHEME_DIR = Path(__file__).parent

# ทั้ง 5 fold ของต้นฉบับ (ไว้อ้างอิง/debug เทียบกับ oos_account.csv เท่านั้น)
ALL_FOLDS = [
    ("2000-01-01", "2013-12-31", "2014-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2002-01-01", "2015-12-31", "2016-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2004-01-01", "2017-12-31", "2018-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2006-01-01", "2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2008-01-01", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2026-07-29"),
]

# Fold ที่ใช้ deploy จริงตอนนี้ = fold ล่าสุด (index 5 เพื่อให้ตรงกับชื่อไฟล์ best_model.zip เดิม)
ACTIVE_FOLD_ID = 5
TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = ALL_FOLDS[ACTIVE_FOLD_ID - 1]

MODELS_DIR = SCHEME_DIR / "models"
STATE_DIR = SCHEME_DIR / "state"
SCALING_STATS_PATH = STATE_DIR / "scaling_stats.json"

NOTES = (
    f"Sliding Window fold {ACTIVE_FOLD_ID}: Train [{TRAIN_START}:{TRAIN_END}] "
    f"(กว้าง 14 ปีคงที่) -> Val [{VAL_START}:{VAL_END}] -> Test [{TEST_START}:{TEST_END}]"
)
