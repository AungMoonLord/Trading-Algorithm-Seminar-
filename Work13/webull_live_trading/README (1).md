# คู่มือรันระบบเทรด Paper Trade (Webull) ด้วยโมเดล PPO / A2C / SAC / TD3

คู่มือนี้เขียนสำหรับ **Windows + conda** อ่านตามลำดับทีละข้อได้เลย ไม่ต้องมีพื้นฐานโค้ด

---

## 0) ระบบนี้ทำอะไร (อ่านก่อน 1 นาที)

1. ดึงราคาหุ้น 5 ตัว (SPY, QQQ, DIA, TLT, GLD) ล่าสุด
2. คำนวณตัวชี้วัดเทคนิคของวันนี้
3. ดูสถานะพอร์ตปัจจุบันของคุณใน Webull (บัญชี Paper Trade)
4. ป้อนข้อมูลทั้งหมดให้โมเดล AI ที่เทรนไว้แล้ว → โมเดลบอกว่า "อยากถือแต่ละตัวกี่ %"
5. คำนวณว่าต้อง ซื้อ/ขาย อะไร กี่หุ้น เพื่อให้พอร์ตเป็นไปตามนั้น
6. **ถ้าใส่ `--dry-run` = แค่แสดงผลให้ดู ไม่ส่งคำสั่งจริง**
   **ถ้าใส่ `--execute` = ส่งคำสั่งจริงเข้าบัญชี Paper Trade**

สคริปต์ **รันครั้งเดียวแล้วจบ** ไม่ได้รันค้างเบื้องหลัง ถ้าคุณไม่สั่ง มันจะไม่ทำอะไรเอง

---

## 1) สถานะปัจจุบัน — อะไรใช้ได้แล้ว / ยังไม่ได้

| ส่วน | สถานะ |
|---|---|
| Sliding Window + PPO (fold 5) | ✅ มีโมเดลแล้ว ใช้ได้ |
| Sliding Window + A2C / SAC / TD3 | ⏳ รอไฟล์ `best_model.zip` ของ fold 5 จากคุณ |
| Expanding Window / Single Holdout | ⛔ ยังใช้ไม่ได้ รอ notebook และโมเดลจริง (ค่าใน config เป็นค่าเดา) |
| โค้ดคำนวณ (feature / โมเดล / risk) | ✅ ทดสอบกับข้อมูลจำลองแล้วผ่าน |
| ต่อ Webull จริง / ดึง yfinance จริง | ❓ **ผู้เขียนยังไม่เคยรันจริง** (เครื่องทดสอบไม่มีอินเทอร์เน็ตออกนอก) ต้องลองบนเครื่องคุณ |
| ส่วนแปลงข้อมูลบัญชี Webull → ตัวเลขให้โมเดล | ⛔ ยังไม่เขียน ต้องดู response จริงก่อน (ดูขั้นตอนที่ 8) |

> จึงคาดว่ารอบแรกจะ **หยุดด้วย error `NotImplementedError` ที่ตั้งใจใส่ไว้** ซึ่งเป็นเรื่องปกติ ไม่ใช่ความผิดพลาดของคุณ

---

## 2) เตรียมของก่อนเริ่ม

- [ ] Anaconda / Miniconda ติดตั้งแล้ว
- [ ] อินเทอร์เน็ต
- [ ] Webull Paper Trade: **App Key, App Secret, Account ID**
- [ ] แตกไฟล์ `webull_live_trading.zip` ไว้ เช่น `C:\Users\ชื่อคุณ\webull_live_trading`

⚠️ **ห้ามส่ง App Key / App Secret ให้ใครหรือโพสต์ในแชท** ใส่ในเครื่องตัวเองเท่านั้น

---

## 3) ขั้นตอนติดตั้ง (ทำครั้งเดียว)

เปิด **Anaconda Prompt** (ค้นหาในเมนู Start)

### 3.1 สร้าง environment
```
conda create -n webull_trading python=3.11 -y
conda activate webull_trading
```
สำเร็จเมื่อหน้าจอขึ้น `(webull_trading)` นำหน้าบรรทัด

### 3.2 ติดตั้งไลบรารี (ใช้เวลาหลายนาที เพราะ torch ไฟล์ใหญ่)
```
pip install yfinance pandas_datareader ta stable-baselines3 gymnasium torch
pip install --upgrade webull-openapi-python-sdk
```

### 3.3 เข้าโฟลเดอร์โปรเจกต์
```
cd C:\Users\ชื่อคุณ\webull_live_trading
```
(แก้ path ให้ตรงกับที่แตกไฟล์ไว้ ถ้าอยู่ไดรฟ์อื่นให้พิมพ์ `D:` ก่อน)

---

## 4) ตั้งค่า API Key (ทำครั้งเดียว conda จำให้)

พิมพ์ทีละบรรทัด แทน `xxxxxxxx` ด้วยค่าจริงของคุณ:
```
conda env config vars set WEBULL_APP_KEY=xxxxxxxx
conda env config vars set WEBULL_APP_SECRET=xxxxxxxx
conda env config vars set WEBULL_ACCOUNT_ID=xxxxxxxx
conda env config vars set WEBULL_ENV=sandbox
```
**สำคัญ:** ต้อง "ปิด-เปิด" environment ก่อน ค่าถึงจะมีผล:
```
conda deactivate
conda activate webull_trading
```
ตรวจว่าบันทึกแล้ว: `conda env config vars list`

`WEBULL_ENV=sandbox` คือโหมดจำลอง (Paper Trade) **ห้ามเปลี่ยนเป็น production**

---

## 5) ทดสอบต่อ Webull

```
python -c "from webull_bridge import WebullBridge, WebullCredentials; b=WebullBridge(WebullCredentials.from_env()); print(b.get_account_summary()); print(b.get_positions())"
```

| ผลลัพธ์ | ความหมาย |
|---|---|
| ขึ้นข้อมูลบัญชี (ตัวเลข/ข้อความ) | ✅ เชื่อมต่อสำเร็จ |
| `KeyError: 'WEBULL_APP_KEY'` | ยังไม่ได้ `deactivate` แล้ว `activate` ใหม่ (กลับไปข้อ 4) |
| error เรื่อง auth / signature / 401 | key, secret หรือ account id พิมพ์ผิด |
| error เรื่อง connection / timeout | อินเทอร์เน็ต หรือไฟร์วอลล์ที่ทำงานบล็อก |

**เก็บข้อความที่ print ออกมาไว้** ต้องใช้ในข้อ 8

---

## 6) สร้าง scaling stats (ทำครั้งเดียวต่อ scheme)

โมเดลถูกเทรนด้วยข้อมูลที่ "ปรับสเกล" แล้ว ตอนใช้จริงต้องปรับสเกลเหมือนเดิม ตัวเลขชุดนี้ไม่ได้ถูกบันทึกไว้ใน notebook เดิม จึงต้องคำนวณใหม่:
```
python fetch_scaling_stats.py --scheme sliding_window
```
- ใช้เวลาหลายนาที (ดึงข้อมูลย้อนหลังตั้งแต่ปี 1996)
- สำเร็จเมื่อเห็นตาราง mean/std และไฟล์ `schemes\sliding_window\state\scaling_stats.json`

**ข้อควรตรวจ:** ค่าจะเหมือนตอนเทรนก็ต่อเมื่อ yfinance ให้ราคาย้อนหลังเหมือนเดิม ควรเทียบผลย้อนหลังกับ `oos_account.csv` ของ fold 5 ก่อนไว้ใจเต็มที่ (บอกผู้ช่วยได้ถ้าอยากให้เขียนสคริปต์เทียบให้)

---

## 7) รันแบบ dry-run (ยังไม่ส่งคำสั่งจริง)

```
python run_live.py --scheme sliding_window --algo ppo --dry-run
```
(ไม่ต้องพิมพ์ `--dry-run` ก็ได้ เพราะเป็นค่าเริ่มต้นอยู่แล้ว)

**รอบแรกจะเกิดขึ้นแบบนี้:**
1. ต่อ Webull สำเร็จ
2. print `Account summary (raw): ...` และ `Positions (raw): ...`
3. หยุดด้วย `NotImplementedError: เติม mapping ...` ← **ตั้งใจให้หยุดตรงนี้**

ถ้าเจอ `[SKIP] ... ไม่พบโมเดล` แปลว่าไฟล์โมเดลไม่ได้อยู่ในตำแหน่งที่ถูกต้อง (ดูข้อ 10)

---

## 8) ส่งผลลัพธ์กลับมาให้ผู้ช่วยเขียนส่วนที่เหลือ

คัดลอกข้อความ `Account summary (raw)` และ `Positions (raw)` จากข้อ 5 หรือ 7 มาให้ดู
(ลบตัวเลขเงิน/จำนวนหุ้นได้ ขอแค่ **ชื่อ field**) ผู้ช่วยจะเขียนโค้ด 3 ค่าที่ขาดอยู่:
`current_equity`, `current_position_values`, `latest_prices` แล้วรอบต่อไปจะรันจบครบ

พอรันจบแบบ dry-run จะเห็น:
```
target_weights : น้ำหนักที่โมเดลอยากได้ (CASH, SPY, QQQ, DIA, TLT, GLD)
current_weights: น้ำหนักที่ถืออยู่ตอนนี้
orders:         รายการซื้อ/ขายที่จะส่ง
(dry-run — ไม่ได้ส่ง order จริง)
```

---

## 9) ส่งคำสั่งจริงเข้า Paper Trade

ทำเมื่อ dry-run ผ่านและตัวเลขดูสมเหตุสมผลเท่านั้น:
```
python run_live.py --scheme sliding_window --algo ppo --execute
```
คำสั่งเป็นแบบ `MARKET_ON_OPEN` = ซื้อ/ขายตอนตลาดสหรัฐฯ **เปิดวันถัดไป** ตรงกับวิธีที่โมเดลถูกเทรน (ตัดสินใจจากราคาปิดวันนี้ เข้าที่ราคาเปิดพรุ่งนี้) ตรวจคำสั่งได้ในแอพ Webull (โหมด Paper)

---

## 10) วางไฟล์โมเดลตรงไหน

```
schemes\sliding_window\models\ppo\fold_5\best_model.zip   ← มีแล้ว
schemes\sliding_window\models\a2c\fold_5\best_model.zip   ← วางเอง
schemes\sliding_window\models\sac\fold_5\best_model.zip   ← วางเอง
schemes\sliding_window\models\td3\fold_5\best_model.zip   ← วางเอง
```
- แต่ละ algo ใช้ **แค่ fold 5** เทรดจริง (fold 1-4 คือการทดสอบย้อนหลัง เก็บไว้เฉยๆ ได้)
- ที่มาของไฟล์: `production_v3_1_results\fold_5\best_model.zip` ของแต่ละ algo
- สลับ algo แค่เปลี่ยน `--algo ppo` เป็น `a2c` / `sac` / `td3`

รันทุกตัวที่พร้อมทีเดียว (ตัวที่ไม่มีไฟล์จะข้ามเอง):
```
python run_all_models.py --dry-run
```
⚠️ แต่ละโมเดลถือเป็นกลยุทธ์อิสระ ถ้าสองตัวสั่งสวนกัน (ตัวหนึ่งซื้อ SPY อีกตัวขาย SPY) ระบบจะส่งทั้งสองคำสั่ง ไม่หักลบกันให้

---

## 11) หยุดโปรแกรมได้ไหม

- กด **`Ctrl + C`** ใน Anaconda Prompt หยุดได้ทันที
- โหมด `--dry-run` หยุดตอนไหนก็ปลอดภัย (ไม่ได้ส่งอะไร)
- โหมด `--execute` ถ้ากดหยุดกลางทาง คำสั่งที่ส่งไปแล้วจะ **ยังอยู่ใน Webull** ต้องเข้าแอพไปยกเลิกเอง

---

## 12) รันตอนไหนดี

โมเดลออกแบบให้รัน **วันละครั้ง หลังตลาดสหรัฐฯ ปิด** (ปิด 16:00 น. เวลานิวยอร์ก ≈ **03:00–04:00 น. เวลาไทย**) และก่อนตลาดเปิดวันถัดไป

ตอนนี้ **ไม่มีการตั้งเวลาอัตโนมัติ** สคริปต์รันเมื่อคุณพิมพ์คำสั่งเท่านั้น ถ้าอยากให้รันเองทุกวัน บน Windows ใช้ **Task Scheduler** (ไม่ใช่ cron) บอกผู้ช่วยให้เขียนไฟล์ `.bat` + ขั้นตอนตั้งค่าให้ได้ แนะนำให้รันด้วยมือสัก 1-2 สัปดาห์ก่อน

**ทุกครั้งที่กลับมารัน** (เปิดเครื่องใหม่):
```
conda activate webull_trading
cd C:\Users\ชื่อคุณ\webull_live_trading
python run_live.py --scheme sliding_window --algo ppo --dry-run
```
ไม่ต้องตั้ง key ซ้ำ

---

## 13) แก้ปัญหาเบื้องต้น

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `'conda' is not recognized` | เปิดผิดโปรแกรม ให้ใช้ Anaconda Prompt |
| `ModuleNotFoundError: No module named ...` | ยังไม่ `conda activate webull_trading` หรือติดตั้งไลบรารีไม่ครบ (ข้อ 3.2) |
| `KeyError: 'WEBULL_APP_KEY'` | ข้อ 4: ต้อง deactivate แล้ว activate ใหม่ |
| `[SKIP] ... ไม่พบ scaling_stats.json` | ยังไม่ได้ทำข้อ 6 |
| `[SKIP] ... ไม่พบโมเดล` | ไฟล์ไม่อยู่ใน `models\<algo>\fold_5\` ตรวจชื่อโฟลเดอร์และชื่อไฟล์ |
| `NotImplementedError` | ตั้งใจให้หยุด ทำข้อ 8 |
| yfinance ดึงข้อมูลไม่ได้ / ว่างเปล่า | อินเทอร์เน็ต หรือ Yahoo จำกัดชั่วคราว รอสักพักแล้วลองใหม่ |
| ค่าที่โมเดลตอบดูแปลกมาก | scaling stats อาจไม่ตรงกับตอนเทรน (ดูข้อควรตรวจในข้อ 6) |

---

## 14) ข้อควรรู้ด้านความเสี่ยง

- ใช้กับ **Paper Trade เท่านั้น** ห้ามเปลี่ยนเป็นเงินจริงจนกว่าจะตรวจสอบผลอย่างจริงจัง
- โมเดล fold 5 เทรนด้วยข้อมูลถึงปี 2021 และเป็นโมเดลวิจัย ยังไม่ได้พิสูจน์กับสภาพตลาดจริง/slippage ถ้าใช้ยาวควรมีแผนเทรนใหม่เป็นระยะ
- ผลย้อนหลัง (`oos_account.csv`) ไม่รับประกันผลในอนาคต
- เอกสารนี้เป็นข้อมูลทางเทคนิค ไม่ใช่คำแนะนำการลงทุน

---

## ภาคผนวก: โครงสร้างไฟล์

```
run_live.py           รันทีละ scheme + algo
run_all_models.py     รันรวมทุกตัวที่พร้อม
fetch_scaling_stats.py สร้างค่าปรับสเกล (ข้อ 6)
common.py             คำนวณ feature / risk / observation
live_inference.py     โมเดล → น้ำหนัก → รายการซื้อขาย
webull_bridge.py      เชื่อม Webull
state_manager.py      จำสถานะ risk engine ข้ามวัน
schemes\<scheme>\config.py   วันที่ train/val/test ของแต่ละวิธี
schemes\<scheme>\models\     ไฟล์โมเดล
schemes\<scheme>\state\      ไฟล์ที่ระบบสร้างเอง (scaling stats, สถานะรายวัน)
```
