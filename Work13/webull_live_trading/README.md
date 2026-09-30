# Webull Paper-Trading Bridge — Sliding Window / Expanding Window / Single Holdout

โครงสร้างใหม่รองรับ 3 วิธีแบ่ง train/val/test (scheme) แต่ละ scheme มี
model/scaling-stats/state ของตัวเอง แยกกันเด็ดขาด ใช้ feature engineering +
risk engine ชุดเดียวกัน (เพราะ 4 notebook ที่ส่งมายืนยันแล้วว่าใช้ pipeline
เดียวกัน 100%)

```
common.py             feature engineering / risk engine / obs builder (ใช้ร่วมทุก scheme+algo)
state_manager.py       persist risk-engine state ข้ามวัน (แยกไฟล์ตาม scheme+algo)
live_inference.py      decide(): obs -> model -> target weights -> orders
webull_bridge.py       wrapper รอบ webull-openapi-python-sdk (sandbox)
fetch_scaling_stats.py สร้าง scaling_stats.json ของ scheme ที่เลือก (รันครั้งเดียวต่อ scheme)
run_live.py             ข้อ 1-3: รันทีละ scheme+algo (--scheme ... --algo ...)
run_all_models.py       ข้อ 4: รันรวมทุก scheme+algo ที่มีโมเดลพร้อมแล้ว

schemes/
  sliding_window/       ✅ พร้อมใช้จริง — ตรงกับ Sliding_Window_*.ipynb ที่ส่งมา
    config.py             fold 5: Train 2008-2021 / Val 2022-23 / Test 2024-26
    models/ppo/best_model.zip   (มีแล้ว จากที่อัปโหลด production_v3_1_results.zip)
    models/a2c|sac|td3/         (ว่าง — รอไฟล์จากคุณ)
    state/

  expanding_window/     ⚠️ PLACEHOLDER — รอ notebook จริงจากคุณ
    config.py             ค่า train/val/test ที่ผมเดาไว้ก่อนตามธรรมเนียมทั่วไป
    models/{ppo,a2c,sac,td3}/   (ว่างทั้งหมด)
    state/

  single_holdout/       ⚠️ PLACEHOLDER — รอ notebook จริงจากคุณ
    config.py             ค่า train/val/test ที่ผมเดาไว้ก่อน
    models/{ppo,a2c,sac,td3}/   (ว่างทั้งหมด)
    state/
```

## ตอบ 4 ข้อที่ถาม

1. **Expanding Window** → `python run_live.py --scheme expanding_window --algo ppo --dry-run`
   ตอนนี้จะ `[SKIP]` เพราะยังไม่มีไฟล์โมเดล/scaling stats (คุณบอกว่าจะส่ง
   notebook มาทีหลัง) — โครงสร้างพร้อมแล้ว พอมีไฟล์มาก็ใส่ตามขั้นตอนด้านล่าง
2. **Sliding Window** → `python run_live.py --scheme sliding_window --algo ppo --dry-run`
   ✅ ใช้งานได้จริงตอนนี้ (มีโมเดล PPO fold 5 อยู่แล้ว)
3. **Single Holdout** → `python run_live.py --scheme single_holdout --algo ppo --dry-run`
   เหมือนข้อ 1 ยัง `[SKIP]` รอไฟล์จากคุณ
4. **รวมทุก model** → `python run_all_models.py --dry-run`
   ลูปทุก (scheme × algo) ที่เจอไฟล์โมเดลจริง ตัวไหนไม่มีจะข้ามอัตโนมัติ
   (ตอนนี้จะรันแค่ `sliding_window/ppo` ตัวเดียว ตัวอื่น skip หมด)

## Setup

```bash
pip install yfinance pandas_datareader ta stable-baselines3 gymnasium torch
pip install --upgrade webull-openapi-python-sdk
```

```bash
export WEBULL_APP_KEY=xxxx        # จาก Paper Trade API key ที่คุณเพิ่งได้มา
export WEBULL_APP_SECRET=xxxx
export WEBULL_ACCOUNT_ID=xxxx
export WEBULL_ENV=sandbox
```

### ทดสอบ credentials ก่อน (สำคัญ)
```python
from webull_bridge import WebullBridge, WebullCredentials
bridge = WebullBridge(WebullCredentials.from_env())
print(bridge.get_account_summary())
print(bridge.get_positions())
```
ผมยืนยันแล้วว่า `WebullBridge` ต่อ network จริงตอนสร้าง object (auth token
check) และ endpoint ที่ผูกไว้ถูกต้อง (`us-openapi-alb.uat.webullbroker.com`)
— เหลือแค่รันบนเครื่องคุณที่มีเน็ตจริง **print(resp) สองบรรทัดนี้ดูก่อนเสมอ**
เพื่อเอา field name จริงไปเติมใน `run_live.py`

### สำหรับ Sliding Window (ใช้ได้จริงตอนนี้)
```bash
python fetch_scaling_stats.py --scheme sliding_window
```
**ตรวจสอบก่อนใช้จริง:** เทียบผลลัพธ์กับ
`production_v3_1_results/fold_5/oos_account.csv` (ไฟล์เดิมที่คุณอัปโหลด) —
รัน scaling+env ย้อนหลังบนช่วง test เดิม (2024-01-01 ถึง 2026-07-29) เทียบ
portfolio_value ว่าใกล้เคียงเดิมไหม ถ้าต่างมาก แปลว่า yfinance คืนราคาย้อนหลัง
ไม่เหมือนตอนเทรน ต้อง debug ก่อน

### สำหรับ Expanding Window / Single Holdout (ยังใช้ไม่ได้จนกว่าจะมี 3 อย่างนี้)
1. ส่ง notebook จริง (หรือแค่ WALK_FORWARD_FOLDS/TRAIN_START-END ที่ใช้จริง) มาให้ผมแก้
   `schemes/expanding_window/config.py` และ `schemes/single_holdout/config.py`
   ให้ตรง (ตอนนี้เป็นค่าเดาไว้ก่อน มี `[PLACEHOLDER]` กำกับชัดเจน)
2. วาง `best_model.zip` ของแต่ละ algo ไว้ที่ `schemes/<scheme>/models/<algo>/best_model.zip`
3. รัน `python fetch_scaling_stats.py --scheme <scheme>`
4. ใช้งานผ่าน `run_live.py` / `run_all_models.py` ได้เหมือน sliding_window ทันที
   (ไม่ต้องแก้โค้ด logic ใดๆ เพิ่ม — ออกแบบให้ใช้ร่วมกันได้แล้ว)

### เติม field mapping ใน `run_live.py` (ทำครั้งเดียว ใช้ได้ทุก scheme/algo)
ตอนนี้มี `raise NotImplementedError(...)` คั่นไว้ในฟังก์ชัน `run_one()` —
เรียก `bridge.get_account_summary()` / `bridge.get_positions()` ดูโครงสร้าง
จริงก่อน แล้วเติม 3 บรรทัดที่ comment ไว้:
```python
current_equity = ...             # net liquidation value
current_position_values = {...}  # {'SPY': mv, 'QQQ': mv, ...}
latest_prices = {...}            # {'SPY': px, ...}
```
แล้วลบบรรทัด `raise NotImplementedError(...)` ออก — ทำจุดเดียว ใช้ได้ทั้ง
`run_live.py` และ `run_all_models.py` (เพราะ `run_all_models.py` เรียก
`run_live.run_one()` ซ้ำ ไม่มีโค้ดซ้ำ)

## รันจริง

```bash
# ทีละตัว
python run_live.py --scheme sliding_window --algo ppo --dry-run
python run_live.py --scheme sliding_window --algo ppo --execute      # ยิงจริงเข้า sandbox

# รวมทุกตัวที่พร้อม
python run_all_models.py --dry-run
python run_all_models.py --execute
python run_all_models.py --schemes sliding_window --algos ppo        # จำกัดเฉพาะบางตัว
```

**`--dry-run` คือค่า default เสมอ ต้องพิมพ์ `--execute` ชัดเจนถึงจะส่ง order จริง**

### จังหวะเวลา
โมเดลตัดสินใจด้วยข้อมูลถึง close วันนี้ แล้ว "เข้าที่ open พรุ่งนี้" (ตรงกับ
`step()` ในต้นฉบับที่ execute ที่ `open_matrix[day+1]`) → โค้ดส่งเป็น order type
`MARKET_ON_OPEN` (มีจริงใน SDK ตรงกับ timing นี้พอดี) ตั้ง cron รันหลังตลาดปิด
ของทุกวันทำการ

### run_all_models.py ไม่ netting ออเดอร์ให้
ถ้า PPO บอกให้ซื้อ SPY แต่ A2C บอกให้ขาย SPY พร้อมกัน สคริปต์จะส่งทั้ง 2 order
จริง (ซื้อ+ขาย) ไม่ได้หักลบกันเอง เพราะแต่ละโมเดลถือเป็นกลยุทธ์อิสระ (ควรคิดว่า
เป็นการแบ่งเงินทดลองเป็นหลาย "sleeve" แยกกัน ไม่ใช่ ensemble เดียว) ถ้าอยากรวม
เป็น ensemble เดียวจริงๆ ต้องออกแบบ position-sizing ชั้นบนอีกที (บอกได้ถ้าอยากได้ต่อ)

## ความเสี่ยงที่ควรรู้ก่อนต่อเงินจริง

- นี่คือโค้ดสำหรับ **paper trading (sandbox)** เท่านั้น อย่าเปลี่ยน
  `WEBULL_ENV=production` จนกว่าจะ validate ผลตรงกับ backtest แล้วจริง ๆ
- Field name ใน `webull_bridge.py` มาจากการอ่าน SDK source + เอกสารทางการ
  (ยืนยันด้วยการรัน `WebullBridge()` จริงจนถึงจุดที่โดน network บล็อก) แต่
  field ของ response (`get_account_summary`/`get_positions`) ยังไม่เคยเห็นจริง
  — print ดูก่อนเชื่อเสมอ
- Expanding Window / Single Holdout ยังเป็นค่าเดา **ห้ามใช้เทรดจริงจนกว่าจะ
  เอาตัวเลขจริงจาก notebook มาแทน**
- Performance ย้อนหลัง (`oos_account.csv`) ไม่ได้การันตีผลตอบแทนอนาคต และ
  ระบบนี้ยังไม่เคยพิสูจน์กับข้อมูล live/slippage จริง
- ไม่ใช่คำแนะนำการลงทุน เป็นข้อมูลทางเทคนิคสำหรับสร้างระบบเทรดเท่านั้น
