"""
telegram_report.py
====================
ระบบจัดรูปแบบข้อความและส่งรายงานเข้า Telegram
รองรับ: รายงานพอร์ตฉบับเต็ม / แจ้งเตือนตลาดปิด / แจ้งเตือนข้อผิดพลาดระบบ (System Error)
"""
import html
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ขนาดกว้างปลอดภัยสำหรับหน้าจอมือถือ (32 ตัวอักษร)
BOX_WIDTH = 32


def money(x):
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "-"


def signed_money(x):
    try:
        value = float(x)
        return f"+${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def line(char="─", width=BOX_WIDTH):
    return char * width


# ============================================================
# 1. ข้อความฉบับย่อ (กรณีตลาดปิด / ยังไม่ถึงเวลาเปิด)
# ============================================================
def build_closed_report(scheme: str, algo: str, status_msg: str) -> str:
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = [
        line("═"),
        " 🔔 MARKET STATUS",
        line("─"),
        f" {status_msg}",
        line("─"),
        f" • Model   : {scheme}/{algo.upper()}",
        f" • Updated : {updated}",
        f" • Status  : ข้ามการส่งออเดอร์",
        line("═")
    ]
    return "\n".join(out)


# ============================================================
# 2. ข้อความฉบับย่อ (กรณีเกิด System Error / โค้ดพัง)
# ============================================================
def build_error_report(scheme: str, algo: str, error_msg: str) -> str:
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = [
        line("═"),
        " 🚨 SYSTEM ERROR ALERT",
        line("─"),
        f" • Model   : {scheme}/{algo.upper()}",
        f" • Updated : {updated}",
        line("─"),
        " [DETAILS]",
        f" {error_msg[:120]}",
        line("─"),
        " • Status  : หยุดทำงานชั่วคราว",
        line("═")
    ]
    return "\n".join(out)


# ============================================================
# 3. ข้อความฉบับเต็ม (กรณีตลาดเปิดทำการปกติ)
# ============================================================
def build_report(bridge, creds, scheme: str, algo: str, executed: bool, result,
                 new_csv_rows=None) -> str:
    balance = bridge.get_account_summary()
    positions = bridge.get_positions()
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_label = "EXECUTE" if executed else "DRY-RUN"

    out = []
    # --- Header ---
    out.append(line("═"))
    out.append(f" 📊 REPORT ({mode_label})")
    out.append(line("─"))
    out.append(f" • Model   : {scheme}/{algo.upper()}")
    out.append(f" • Updated : {updated}")
    
    # --- Account Summary ---
    out.append(line("─"))
    out.append(" 💰 ACCOUNT SUMMARY")
    out.append(line("─"))
    out.append(f" Net Liq : {money(balance.get('total_net_liquidation_value'))}")
    out.append(f" Cash    : {money(balance.get('total_cash_balance'))}")
    out.append(f" Mkt Val : {money(balance.get('total_market_value'))}")
    out.append(f" Unrl P/L: {signed_money(balance.get('total_unrealized_profit_loss'))}")
    out.append(f" Day P/L : {signed_money(balance.get('total_day_profit_loss'))}")

    # --- Portfolio ---
    out.append(line("─"))
    out.append(" 💼 POSITIONS")
    out.append(line("─"))
    out.append(f"{'Sym':<5}{'Qty':>5}{'Price':>10}{'MktVal':>11}")
    out.append(line("─"))
    
    for p in positions:
        try:
            out.append(
                f"{p['symbol']:<5}"
                f"{int(float(p['quantity'])):>5,}"
                f"{money(p['last_price']):>10}"
                f"{money(p['market_value']):>11}"
            )
        except (KeyError, ValueError):
            out.append(f"(Err: {p.get('symbol', '?')})")
            
    if not positions:
        out.append(" (ไม่มีถือครอง Position)")

    # --- Orders ---
    out.append(line("─"))
    out.append(" 📋 ORDERS")
    out.append(line("─"))
    
    if result.orders:
        for o in result.orders:
            out.append(f" • {o.side:<4} {o.approx_qty:>4,} {o.ticker:<5} "
                       f"({o.current_weight:.1%}->{o.target_weight:.1%})")
    else:
        out.append(" • ไม่มีคำสั่งซื้อขาย (No-Trade)")
        
    out.append(f" • Turnover : {result.turnover:.2%}")

    # --- System Status ---
    out.append(line("─"))
    out.append(" ⚙️ STATUS")
    out.append(line("─"))
    try:
        open_orders = bridge.list_open_orders()
        n_open = len(open_orders.get("data", open_orders)) if isinstance(open_orders, dict) else len(open_orders)
    except Exception:
        n_open = "?"
        
    out.append(f" Open Orders : {n_open}")
    out.append(f" Margin Call : {'YES' if balance.get('open_margin_calls') else 'None'}")
    out.append(line("═"))

    return "\n".join(out)


# ============================================================
# Telegram API Sender
# ============================================================
def is_configured() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)


def send(text: str) -> bool:
    if not is_configured():
        print("[telegram] ยังไม่ได้ตั้งค่า TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ใน .env")
        return False

    safe_text = html.escape(text)
    body = f"<pre>{safe_text}</pre>"
    
    if len(body) > 4000:
        safe_truncated = html.escape(text[:3900])
        body = f"<pre>{safe_truncated}\n...(ตัดข้อความ)</pre>"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": body, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[telegram] ส่งไม่สำเร็จ: HTTP {resp.status_code} — {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"[telegram] ส่งไม่สำเร็จ: {e}")
        return False