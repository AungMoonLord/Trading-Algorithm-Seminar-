"""
telegram_report.py
====================
ไฟล์เดียวจบสำหรับ Telegram: ประกอบข้อความ (เอา format จาก dashboard script
ของคุณมาใช้ตรงๆ) + ส่งเข้า Telegram

ตั้งค่าใน .env (เพิ่มต่อจาก WEBULL_* เดิม):
    TELEGRAM_BOT_TOKEN=123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    TELEGRAM_CHAT_ID=123456789

วิธีได้ 2 ค่านี้ (ทำครั้งเดียว):
1. คุยกับ @BotFather ใน Telegram -> /newbot -> ได้ token มาเป็น TELEGRAM_BOT_TOKEN
2. ทักบอทที่สร้าง ส่งข้อความอะไรก็ได้ 1 ข้อความไปหาก่อน
3. เปิด https://api.telegram.org/bot<TOKEN>/getUpdates ในเบราว์เซอร์
   หา "chat":{"id": ...} เอาเลขนั้นมาใส่ TELEGRAM_CHAT_ID

ไม่ตั้งค่าไว้ก็ไม่เป็นไร โปรแกรมจะข้ามการส่งเงียบๆ ไม่ error (ไม่กระทบการเทรด)
"""
import os
from pathlib import Path
import html  # <--- 1. เพิ่ม import html ไว้ด้านบนสุดของไฟล์

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# ============================================================
# FORMAT FUNCTIONS (เอามาจาก dashboard script ของคุณตรงๆ)
# ============================================================
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


def line(char="-", width=50):
    return char * width


# ============================================================
# ประกอบข้อความ — โครงเดียวกับ dashboard script ของคุณ
# ============================================================
def build_report(bridge, creds, scheme: str, algo: str, executed: bool, result,
                  new_csv_rows=None) -> str:
    """
    bridge, creds: ตัวเดียวกับที่ใช้ใน run_live.py อยู่แล้ว
    result: DecisionResult จาก live_inference.decide()
    new_csv_rows: list ที่ได้จาก trade_logger.log_order_history_to_csv() (ถ้ามี)
    """
    from datetime import datetime

    balance = bridge.get_account_summary()
    positions = bridge.get_positions()
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    environment = "SANDBOX / PAPER" if creds.sandbox else "LIVE"

    out = []
    out.append(line("="))
    out.append(f"{scheme} / {algo.upper()}  |  {'EXECUTE (ส่งจริง)' if executed else 'DRY-RUN (ตัวอย่าง)'}")
    out.append(line("="))
    out.append(f"Updated      : {updated}")
    out.append(f"Environment  : {environment}")
    out.append(f"Currency     : {balance.get('total_asset_currency', '-')}")
    out.append("")

    out.append(line())
    out.append("ACCOUNT SUMMARY")
    out.append(line())
    out.append(f"Net Liquidation Value : {money(balance.get('total_net_liquidation_value'))}")
    out.append(f"Cash Balance          : {money(balance.get('total_cash_balance'))}")
    out.append(f"Market Value          : {money(balance.get('total_market_value'))}")
    out.append(f"Unrealized P/L        : {signed_money(balance.get('total_unrealized_profit_loss'))}")
    out.append(f"Day P/L               : {signed_money(balance.get('total_day_profit_loss'))}")
    out.append(f"Maintenance Margin    : {money(balance.get('maintenance_margin'))}")
    out.append("")

    out.append(line())
    out.append("PORTFOLIO")
    out.append(line())
    out.append(f"{'Symbol':<8}{'Qty':>8}{'Price':>12}{'Mkt Value':>14}{'P/L':>12}")
    out.append(line())
    for p in positions:
        try:
            out.append(
                f"{p['symbol']:<8}{int(float(p['quantity'])):>8,}"
                f"{money(p['last_price']):>12}{money(p['market_value']):>14}"
                f"{signed_money(p.get('unrealized_profit_loss')):>12}"
            )
        except (KeyError, ValueError):
            out.append(f"(เจอ field ที่ไม่รู้จักใน position: {p})")
    if not positions:
        out.append("(ไม่มี position อยู่)")
    out.append("")

    out.append(line())
    out.append(f"{'คำสั่งรอบนี้' if executed else 'คำสั่งที่จะส่ง (ถ้า --execute)'}")
    out.append(line())
    if result.orders:
        for o in result.orders:
            out.append(f"  {o.side:<4} {o.approx_qty:>6,} {o.ticker:<5} "
                       f"(target {o.target_weight:.1%} <- {o.current_weight:.1%})")
    else:
        out.append("  ไม่มีคำสั่งซื้อขาย (น้ำหนักใกล้เคียงเดิม ไม่เกิน no-trade band)")
    out.append(f"  Turnover: {result.turnover:.2%}  |  Risk scalar: {result.scalar:.2f}")

    if new_csv_rows:
        out.append("")
        out.append(f"บันทึก CSV แล้ว {len(new_csv_rows)} order ใหม่")

    out.append("")
    out.append(line())
    out.append("STATUS")
    out.append(line())
    try:
        open_orders = bridge.list_open_orders()
        n_open = len(open_orders.get("data", open_orders)) if isinstance(open_orders, dict) else len(open_orders)
    except Exception:
        n_open = "?"
    out.append(f"Open Orders : {n_open}")
    out.append(f"Margin Call : {'YES' if balance.get('open_margin_calls') else 'None'}")
    out.append("Account     : Connected")
    out.append(line("="))

    return "\n".join(out)


# ============================================================
# ส่งเข้า Telegram
# ============================================================
def is_configured() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)




def send(text: str) -> bool:
  if not is_configured():
    print(
        "[telegram] ยังไม่ได้ตั้งค่า TELEGRAM_BOT_TOKEN /"
        " TELEGRAM_CHAT_ID ใน .env — ข้ามการแจ้งเตือน"
    )
    return False

  # <--- 2. แปลงเครื่องหมายพิเศษเช่น < และ > ให้ปลอดภัยสำหรับ HTML
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
      print(
          f"[telegram] ส่งไม่สำเร็จ: HTTP {resp.status_code} —"
          f" {resp.text}"
      )
      return False
    return True
  except Exception as e:
    print(f"[telegram] ส่งไม่สำเร็จ: {e}")
    return False
