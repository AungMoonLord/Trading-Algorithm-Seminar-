"""
webull_bridge.py
=================
Wrapper รอบ official "webull-openapi-python-sdk" (package บน PyPI ชื่อ
webull-openapi-python-sdk แต่ import เป็น `webull` เฉยๆ — คนละตัวกับ unofficial
tedchou12/webull)

ติดตั้ง:
    pip3 install --upgrade webull-openapi-python-sdk python-dotenv

ผมติดตั้งแพ็กเกจนี้จริงและอ่าน source code (webull/trade/, webull/core/) เพื่อ
ยืนยัน method/field names ด้านล่าง + เทียบกับตัวอย่าง JSON บนหน้า docs ทางการ
(developer.webull.com/apis/docs/trade-api/trade) แล้ว

Base URL (ยืนยันจาก developer.webull.com/apis/docs/trade-api/trade):
    Production : https://api.webull.com/
    Test/UAT   : http://us-openapi-alb.uat.webullbroker.com/   (สังเกตเป็น http ไม่ใช่ https)

Credentials:
    อ่านจากไฟล์ .env ในโฟลเดอร์เดียวกับไฟล์นี้ (ดู .env.example) หรือจาก
    environment variables ตรงๆ ก็ได้ (เช่นที่ตั้งผ่าน conda env config vars set)
    .env จะ override ค่าที่ตั้งไว้ใน conda ถ้ามีทั้งคู่
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # โหลดค่าจาก .env เข้าเป็น environment variables

from dataclasses import dataclass
from typing import Optional

SANDBOX_HOST = "api.sandbox.webull.com"   # ยืนยันจากเอกสารทางการ
PRODUCTION_HOST = "api.webull.com"


@dataclass
class WebullCredentials:
    app_key: str
    app_secret: str
    account_id: str
    region_id: str = "us"
    sandbox: bool = True

    @classmethod
    def from_env(cls) -> "WebullCredentials":
        return cls(
            app_key=os.environ["WEBULL_APP_KEY"],
            app_secret=os.environ["WEBULL_APP_SECRET"],
            account_id=os.environ["WEBULL_ACCOUNT_ID"],
            region_id=os.environ.get("WEBULL_REGION", "us"),
            sandbox=os.environ.get("WEBULL_ENV", "sandbox").lower() != "production",
        )


class WebullBridge:
    def __init__(self, creds: WebullCredentials):
        from webull.core.client import ApiClient
        from webull.trade.trade_client import TradeClient

        self.creds = creds
        self.client = ApiClient(creds.app_key, creds.app_secret, creds.region_id)
        if creds.sandbox:
            # ผูก host ของ Test/UAT environment เข้ากับ region นี้แทนค่า default (production)
            self.client.add_endpoint(creds.region_id, SANDBOX_HOST, api_type="api")
        self.trade = TradeClient(self.client)

    # ---------------------------------------------------------------- account
    def get_account_list(self) -> dict:
        """ใช้เช็คว่า key/secret ตอนนี้เข้าถึง account_id ไหนได้บ้าง — เรียกก่อน
        get_account_summary() ทุกครั้งที่เจอ 403 ACCOUNT_ACCESS_DENIED เพราะมักแปลว่า
        account_id เดิมไม่ตรงกับที่ key นี้เข้าถึงได้แล้ว (เช่น หลัง reset บัญชี)"""
        resp = self.trade.account_v2.get_account_list()
        return resp.json() if hasattr(resp, "json") else resp

    def get_account_summary(self) -> dict:
        """คืน raw response ของ get_account_balance — print(resp.json()) ดูก่อนใช้จริง
        เพื่อเช็ค field ที่แน่นอน (เช่น net liquidation value / cash balance)"""
        resp = self.trade.account_v2.get_account_balance(self.creds.account_id)
        return resp.json() if hasattr(resp, "json") else resp

    def get_positions(self) -> dict:
        """คืน raw response ของ get_account_position — print(resp.json()) ดูก่อนใช้จริง"""
        resp = self.trade.account_v2.get_account_position(self.creds.account_id)
        return resp.json() if hasattr(resp, "json") else resp

    # ------------------------------------------------------------------ order
    def submit_order(
        self, client_order_id: str, symbol: str, side: str, order_type: str,
        quantity: int, limit_price: Optional[float] = None,
        time_in_force: str = "DAY", extended_hours: bool = False,
    ) -> dict:
        """
        side: 'BUY' | 'SELL'
        order_type: 'MARKET' | 'LIMIT' | 'MARKET_ON_OPEN' | ... (ดู webull/trade/common/order_type.py)
        รูปแบบ order dict อ้างอิงจากตัวอย่างจริงในเอกสาร
        developer.webull.com/apis/docs/trade-api/trade
        """
        order = {
            "client_order_id": client_order_id,
            "instrument_type": "EQUITY",
            "symbol": symbol,
            "market": "US",
            "side": side,
            "order_type": order_type,
            "quantity": str(quantity),
            "support_trading_session": "EXTENDED" if extended_hours else "CORE",
            "entrust_type": "QTY",
            "time_in_force": time_in_force,
            "combo_type": "NORMAL",
        }
        if order_type in ("LIMIT", "STOP_LOSS_LIMIT", "ENHANCED_LIMIT", "AT_AUCTION_LIMIT", "ODD_LOT_LIMIT"):
            assert limit_price is not None, f"{order_type} ต้องระบุ limit_price"
            order["limit_price"] = str(limit_price)

        resp = self.trade.order_v3.place_order(self.creds.account_id, [order])
        return resp.json() if hasattr(resp, "json") else resp

    def submit_market_on_open_order(self, client_order_id: str, symbol: str, side: str, quantity: int) -> dict:
        """ชื่อฟังก์ชันยังคงเดิมไว้เพื่อให้ run_live.py ไม่ต้องแก้ แต่ส่งเป็น "MARKET" ธรรมดา
        เพราะ MARKET_ON_OPEN เป็น order type สำหรับบัญชี Institutional เท่านั้น
        (ยืนยันจากเอกสาร Webull: 'Execute at the opening price (institutional only)')
        บัญชี Individual อย่างเราใช้ไม่ได้ — ส่ง MARKET แทน ซึ่งถ้าส่งตอนตลาดปิด
        จะถูก queue ไว้แล้ว execute ตอนตลาดเปิดรอบถัดไปโดยอัตโนมัติอยู่แล้ว
        (ยืนยันจาก order ที่เคย FILLED จริงในบัญชีคุณ)"""
        return self.submit_order(client_order_id, symbol, side, "MARKET", quantity)

    def cancel_order(self, client_order_id: str) -> dict:
        resp = self.trade.order_v3.cancel_order(self.creds.account_id, client_order_id)
        return resp.json() if hasattr(resp, "json") else resp

    def list_open_orders(self) -> dict:
        resp = self.trade.order_v3.list_order_open(self.creds.account_id)
        return resp.json() if hasattr(resp, "json") else resp

    def get_order_history(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
        """ดูประวัติ order ทั้งหมด (รวมที่ถูก reject/cancel ด้วย) ต่างจาก list_open_orders
        ที่เห็นแค่ order ที่ยัง pending อยู่เท่านั้น — ใช้ตัวนี้เพื่อดูว่า order ที่ส่งไป
        เกิดอะไรขึ้นจริง ถ้าไม่ระบุวันที่ จะดึงแค่ 7 วันล่าสุด"""
        resp = self.trade.order_v3.get_order_history(
            self.creds.account_id, start_date=start_date, end_date=end_date
        )
        return resp.json() if hasattr(resp, "json") else resp

    def get_order_detail(self, client_order_id: str) -> dict:
        """เช็คสถานะ order เดี่ยวๆ ด้วย client_order_id ที่เราตั้งเอง (เช่น 'sliding_window_ppo_SPY')"""
        resp = self.trade.order_v3.get_order_detail(self.creds.account_id, client_order_id)
        return resp.json() if hasattr(resp, "json") else resp