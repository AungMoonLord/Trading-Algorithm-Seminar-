# Webull Paper Trading Bot — PPO (Sliding / Expanding / Single Holdout)

ระบบเทรดอัตโนมัติที่เอาโมเดล PPO (เทรนด้วย Stable-Baselines3 บนพอร์ต 5 ETF:
SPY, QQQ, DIA, TLT, GLD) มาต่อกับ **Webull OpenAPI (Sandbox / Paper Trade)**
รองรับโมเดล 3 แบบที่เทรนด้วยวิธีแบ่งข้อมูลต่างกัน ภายใต้ feature engineering
และ risk engine ชุดเดียวกัน

---

## สถานะปัจจุบัน (อัปเดตล่าสุด)

| Scheme | config.py | best_model.zip | สถานะรวม |
|---|---|---|---|
| **Sliding Window** | ✅ ยืนยันจาก notebook จริง | ✅ มีแล้ว (fold 5) | ✅ พร้อมใช้งาน |
| **Expanding Window** | ✅ ยืนยันจาก notebook จริง | ✅ มีแล้ว (fold 5) | ✅ พร้อมใช้งาน |
| **Single Holdout** | ✅ ยืนยันจาก notebook จริง | ✅ มีแล้ว (fold 5) | ✅ พร้อมใช้งาน|

ใช้แค่ **PPO** algorithm เท่านั้น

เชื่อมต่อ Webull Sandbox สำเร็จแล้วจริง มี order FILLED จริงในบัญชี Paper Trade

---

## ภาพรวมระบบทำงานยังไง

```
1. ดึงราคาหุ้น 5 ตัวล่าสุดจาก yfinance + ข้อมูล macro จาก FRED
2. คำนวณ technical indicators (RSI, MACD, EMA ฯลฯ) แล้วปรับสเกล
3. ดึงสถานะพอร์ตปัจจุบันจริงจากบัญชี Webull (เงินสด, หุ้นที่ถืออยู่)
4. ป้อนข้อมูลทั้งหมดให้โมเดล PPO → ได้น้ำหนักพอร์ตที่ "อยากถือ"
5. เทียบกับที่ถืออยู่จริง → คำนวณรายการซื้อ/ขาย
6. --dry-run = แสดงผลอย่างเดียว | --execute = ส่งคำสั่งจริงเข้า Webull
```

---

## โครงสร้างไฟล์

```
common.py                 feature engineering / risk engine / observation builder
                           (ใช้ร่วมกันได้ทุก scheme — ยืนยันแล้วว่า 3 scheme ใช้สูตรเดียวกัน)
state_manager.py           persist risk-engine state ข้ามวัน (แยกไฟล์ตาม scheme)
live_inference.py          decide(): obs → โมเดล → น้ำหนักพอร์ต → รายการซื้อขาย
webull_bridge.py           wrapper รอบ webull-openapi-python-sdk (sandbox)
fetch_scaling_stats.py     สร้างค่าปรับสเกลของ scheme ที่เลือก (รันครั้งเดียวต่อ scheme)
run_live.py                 รันทีละ scheme (--scheme ... --algo ppo)
run_all_models.py           รันรวมทุก scheme ที่มีโมเดลพร้อมแล้ว
.env                        เก็บ API key (ไม่ commit เข้า git)
.env.example                ตัวอย่างไฟล์ .env
.gitignore                  กันไฟล์ลับ/ไฟล์ชั่วคราวหลุดเข้า git

schemes/
  sliding_window/
    config.py                 วันที่ train/val/test (5 fold, หน้าต่างเลื่อน)
    models/ppo/fold_5/best_model.zip
    state/                     scaling_stats.json + live state (สร้างอัตโนมัติ)
  expanding_window/
    config.py                 วันที่ train/val/test (5 fold, train_start คงที่)
    models/ppo/fold_5/best_model.zip
    state/
  single_holdout/
    config.py                 วันที่ train/val/test (split เดียว)
    models/ppo/fold_1/best_model.zip   ← ยังไม่มี ต้องวางเพิ่ม
    state/
```

---

## Setup (ทำครั้งเดียว)

### 1) สร้าง environment
```powershell
conda create -n webull_trading python=3.11 -y
conda activate webull_trading
```

### 2) ติดตั้งไลบรารี
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) ตั้งค่า API key ผ่าน `.env`
สร้างไฟล์ `.env` (ก็อปจาก `.env.example`) ในโฟลเดอร์เดียวกับโค้ด:
```
WEBULL_APP_KEY=xxxxxxxx
WEBULL_APP_SECRET=xxxxxxxx
WEBULL_ACCOUNT_ID=xxxxxxxx
WEBULL_ENV=sandbox
```
**ห้าม commit ไฟล์นี้เข้า git** (มี `.gitignore` กันไว้ให้แล้ว)

### 4) ทดสอบการเชื่อมต่อ
```powershell
python -c "from webull_bridge import WebullBridge, WebullCredentials; b=WebullBridge(WebullCredentials.from_env()); print(b.get_account_summary())"
```
ควรเห็นข้อมูลบัญชี (เช่น `total_net_liquidation_value`) 

---

## การใช้งาน

1.) วางไฟล์โมเดลตรงไหน

```
schemes\expanding_window\models\ppo\fold_5\best_model.zip
schemes\sliding_window\models\ppo\fold_5\best_model.zip
schemes\single_holdout\models\ppo\fold_5\best_model.zip
```
- ใช้ **แค่ fold 5**


2.) สร้าง scaling stats (ทำครั้งเดียวต่อ scheme)
```powershell
## รันทีละ scheme
python fetch_scaling_stats.py --scheme sliding_window  # รันครั้งแรกครั้งเดียวต่อ แต่ละวิธีการเทรน
python fetch_scaling_stats.py --scheme expanding_window
python fetch_scaling_stats.py --scheme single_holdout
```
- สำเร็จเมื่อเห็นตาราง mean/std และไฟล์ `schemes\sliding_window\state\scaling_stats.json`
- เปลี่ยน `sliding_window` เป็น `expanding_window` / `single_holdout` ได้เลย

- ต้องรัน `fetch_scaling_stats.py` บ่อยแค่ไหน — **แค่ครั้งเดียวต่อ scheme(ต่อ 1 วิธีการเทรน)**

`scaling_stats.json` คือค่า mean/std ที่ใช้ปรับสเกล feature ก่อนป้อนเข้าโมเดล
คำนวณจาก **ช่วงวันที่ train ที่ตายตัวแล้ว** ในไฟล์ `config.py` ของแต่ละ scheme
(เช่น Sliding Window fold 5 = Train 2008-01-01 ถึง 2021-12-31) **ไม่ได้ขึ้นกับ
"วันนี้" เลย** ต่อให้รันซ้ำพรุ่งนี้หรือเดือนหน้า ค่าที่ได้ก็เหมือนเดิม
(ตราบใดที่ yfinance ไม่เปลี่ยนราคาย้อนหลัง) รันซ้ำไปก็ไม่มีประโยชน์เพิ่ม แค่
เสียเวลาดึงข้อมูลย้อนหลังตั้งแต่ปี 1996 ใหม่โดยใช่เหตุ

3.) ส่งคำสั่งจริงเข้า Paper Trade
```powershell
python run_live.py --scheme sliding_window --algo ppo #ยังไม่ส่งคำสั่งจริง
python run_live.py --scheme sliding_window --algo ppo --execute   # ส่งจริง

python run_live.py --scheme expanding_window --algo ppo #ยังไม่ส่งคำสั่งจริง
python run_live.py --scheme expanding_window --algo ppo --execute   # ส่งจริง

python run_live.py --scheme single_holdout --algo ppo #ยังไม่ส่งคำสั่งจริง
python run_live.py --scheme single_holdout --algo ppo --execute   # ส่งจริง
```

### รันรวมทุก scheme ที่พร้อม
```powershell
python run_all_models.py --dry-run
python run_all_models.py --execute
```
Scheme ที่ยังไม่มี `best_model.zip` หรือ `scaling_stats.json` จะถูกข้ามอัตโนมัติ
(ไม่ error)

---
