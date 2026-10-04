"""
telegram_csv.py
================
บันทึกประวัติ order ลง CSV แยกตาม scheme/algo — ดัดแปลงจากสคริปต์ที่คุณทำเอง
(trade_data/<algo>/<scheme>/trade_history.csv) กันบันทึกซ้ำด้วย Order_ID เหมือนเดิม

ใช้ตอนหลัง --execute เท่านั้น (dry-run ไม่มี order จริงให้บันทึก)
"""
import csv
from pathlib import Path
from typing import List, Optional


def csv_path(scheme: str, algo: str) -> Path:
    return Path("trade_data") / algo / scheme / "trade_history.csv"


def _load_existing_order_ids(csv_file: Path) -> set:
    existing = set()
    if csv_file.exists():
        with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                oid = row.get("Order_ID")
                if oid:
                    existing.add(oid)
    return existing


def log_order_history_to_csv(scheme: str, algo: str, order_history) -> List[dict]:
    """
    order_history: ผลลัพธ์ดิบจาก bridge.get_order_history()
    คืน list ของ "แถวใหม่" ที่เพิ่งถูกบันทึก (เอาไปใส่ในข้อความ Telegram ได้ด้วย)
    เขียนไฟล์แบบ append — รันซ้ำได้เรื่อยๆ ไม่ซ้ำแถวเดิม
    """
    csv_file = csv_path(scheme, algo)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    existing_order_ids = _load_existing_order_ids(csv_file)
    file_exists = csv_file.exists()
    new_rows: List[dict] = []

    # รองรับทั้ง response ที่เป็น {"data": [...]} และ list ตรงๆ
    orders = order_history.get("data", order_history) if isinstance(order_history, dict) else order_history
    orders = orders or []

    with open(csv_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Order_ID", "Date", "Time", "Symbol", "Side", "Qty", "Price", "Status"])

        for order in orders:
            # บาง response ซ้อน list "orders" อยู่ข้างใน บางอันเป็น trade เดี่ยวๆ เลย — รองรับทั้งคู่
            trades = order.get("orders", [order]) if isinstance(order, dict) else [order]
            for trade in trades:
                order_id = trade.get("order_id") or trade.get("client_order_id") or "-"
                if order_id in existing_order_ids or order_id == "-":
                    continue

                datetime_str = trade.get("filled_time_at", "-")
                if datetime_str and datetime_str != "-":
                    date, time = datetime_str[:10], datetime_str[11:19]
                else:
                    date, time = "-", "-"

                symbol = trade.get("symbol", "-")
                side = trade.get("side", "-")
                quantity = trade.get("filled_quantity", "0") or "0"
                price = trade.get("filled_price", "0") or "0"
                status = trade.get("status", "-")

                writer.writerow([order_id, date, time, symbol, side, int(float(quantity)), price, status])
                existing_order_ids.add(order_id)
                new_rows.append({
                    "order_id": order_id, "date": date, "time": time, "symbol": symbol,
                    "side": side, "qty": int(float(quantity)), "price": price, "status": status,
                })

    return new_rows
