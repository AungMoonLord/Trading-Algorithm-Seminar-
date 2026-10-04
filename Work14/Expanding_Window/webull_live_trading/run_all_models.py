"""
run_all_models.py
==================
รันทุก (scheme, algo) ที่ "มีโมเดลพร้อมแล้วจริง" (เช็คจากไฟล์ best_model.zip
ในเครื่อง — ตัวไหนยังไม่มีไฟล์จะข้ามให้อัตโนมัติ พร้อม print แจ้งว่าข้าม)

    python run_all_models.py --dry-run
    python run_all_models.py --execute
    python run_all_models.py --schemes sliding_window --algos ppo a2c   # เลือกเฉพาะบางตัว

ใช้ Webull connection เดียวกันทุก model (ไม่ต้อง login ใหม่ทุกรอบ) แต่ order/
state ของแต่ละ (scheme, algo) แยกจากกันเด็ดขาด (คนละ client_order_id, คนละ
state file) — ถ้าหลาย model แนะนำ symbol เดียวกันในทิศทางตรงข้ามกัน สคริปต์นี้
**ไม่ได้ netting ให้อัตโนมัติ** จะส่ง order ซ้อนกันตามที่แต่ละโมเดลสั่งจริง —
ถ้าไม่อยากให้เกิดกรณีนี้ ให้รันทีละ (scheme, algo) ด้วย run_live.py แล้วเทียบเอง
ก่อนตัดสินใจส่งจริง
"""
import argparse

import run_live as RL
from webull_bridge import WebullBridge, WebullCredentials

ALL_SCHEMES = ["sliding_window", "expanding_window", "single_holdout"]
ALL_ALGOS = ["ppo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=ALL_SCHEMES, choices=ALL_SCHEMES)
    ap.add_argument("--algos", nargs="+", default=ALL_ALGOS, choices=ALL_ALGOS)
    ap.add_argument("--fetch-end", default="2026-09-26")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    creds = WebullCredentials.from_env()
    bridge = WebullBridge(creds)   # login ครั้งเดียว ใช้ร่วมกันทุก model

    results = {}
    ran, skipped = 0, 0
    for scheme in args.schemes:
        for algo in args.algos:
            res = RL.run_one(scheme, algo, args.execute, args.fetch_end, bridge=bridge)
            if res is None:
                skipped += 1
            else:
                ran += 1
                results[f"{scheme}/{algo}"] = res

    print(f"\n{'='*70}\nสรุป: รันจริง {ran} model, ข้าม {skipped} model (ยังไม่มีโมเดล/scaling stats)\n{'='*70}")
    for name, res in results.items():
        print(f"{name:35s} turnover={res.turnover:.4f}  orders={len(res.orders)}")


if __name__ == "__main__":
    main()
