"""
run_live.py
============
Entrypoint สำหรับรันเทรดจริง — เลือก scheme + algo
รองรับการส่งแจ้งเตือน Telegram รายงานพอร์ต / ตลาดปิด / ข้อผิดพลาดระบบ (System Error)
"""
import argparse
import datetime as dt
import importlib
import sys
import traceback
from datetime import datetime, time, timezone, timedelta
from pathlib import Path

import common as C
import live_inference as LI
import telegram_report as TG
import telegram_csv as TL
from webull_bridge import WebullBridge, WebullCredentials


def check_us_market_status() -> tuple[bool, str]:
    """ตรวจสอบเวลาฝั่ง US Eastern Time (ET) คืนค่า (is_open, status_message)"""
    try:
        tz_et = timezone(timedelta(hours=-4))  # US Eastern Time (EDT)
        now_et = datetime.now(tz_et)
        weekday = now_et.weekday()  # 0=Mon, ..., 5=Sat, 6=Sun
        current_time = now_et.time()

        market_open = time(9, 30)
        market_close = time(16, 0)

        if weekday in (5, 6):
            return False, "⚠️ ตลาดปิดทำการ (วันหยุดเสาร์-อาทิตย์ / นักขัตฤกษ์)"
        elif current_time < market_open:
            return False, "⏰ ตลาดเปิดวันนี้ แต่ยังไม่ถึงเวลาทำการ (รอเปิด 20:30 / 21:30 น.)"
        elif current_time > market_close:
            return False, "🌙 ตลาดปิดทำการแล้วสำหรับวันนี้ (After-Hours)"
        else:
            return True, "🟢 ตลาดเปิดทำการ"
    except Exception:
        return True, "UNKNOWN"


def model_path_for(cfg, algo: str) -> Path:
    return cfg.MODELS_DIR / algo / f"fold_{cfg.ACTIVE_FOLD_ID}" / "best_model.zip"


def parse_account_equity(acct: dict) -> float:
    if "total_net_liquidation_value" not in acct:
        raise KeyError(f"ไม่พบ total_net_liquidation_value ใน account summary: {acct}")
    return float(acct["total_net_liquidation_value"])


def parse_position_values(positions_raw, tickers) -> dict:
    items = positions_raw
    if isinstance(items, dict):
        items = items.get("positions") or items.get("data") or []
    out = {t: 0.0 for t in tickers}
    for it in items:
        sym = it.get("symbol")
        if sym not in out:
            continue
        mv = it.get("market_value")
        if mv is None:
            qty, px = it.get("quantity"), it.get("last_price")
            if qty is None or px is None:
                raise KeyError(f"ไม่รู้จัก field ของ position นี้: {it}")
            mv = float(qty) * float(px)
        out[sym] = float(mv)
    return out


def run_one(scheme: str, algo: str, execute: bool, fetch_end: str, bridge: WebullBridge = None,
            notify: bool = True):
    try:
        cfg = importlib.import_module(f"schemes.{scheme}.config")
        model_path = model_path_for(cfg, algo)
        model_name = f"{scheme}_{algo}"

        if not model_path.exists():
            print(f"[SKIP] {model_name}: ไม่พบโมเดลที่ {model_path}")
            return None
        if not cfg.SCALING_STATS_PATH.exists():
            print(f"[SKIP] {model_name}: ไม่พบ {cfg.SCALING_STATS_PATH} — รัน "
                  f"`python fetch_scaling_stats.py --scheme {scheme}` ก่อน")
            return None

        # ---- 1. ตรวจสอบสถานะตลาดก่อนประมวลผล ----
        is_market_open, market_status_msg = check_us_market_status()
        if not is_market_open:
            print(f"\n[MARKET CLOSED] {market_status_msg}")
            if notify:
                short_text = TG.build_closed_report(scheme, algo, market_status_msg)
                print(short_text)
                TG.send(short_text)
            return None

        print(f"\n{'='*70}\n{model_name}  |  {cfg.NOTES}\n{'='*70}")
        scaling_stats = C.load_scaling_stats(cfg.SCALING_STATS_PATH)

        if bridge is None:
            creds = WebullCredentials.from_env()
            bridge = WebullBridge(creds)

        acct = bridge.get_account_summary()
        positions_raw = bridge.get_positions()

        raw = C.fetch_and_stitch_universe(C.CFG.core_tickers, "2024-01-01", fetch_end)
        feats = C.compute_institutional_features(raw, C.CFG.core_tickers)
        clean = C.clean_and_align_dataset(feats, C.CFG.core_tickers)

        current_equity = parse_account_equity(acct)
        current_position_values = parse_position_values(positions_raw, C.CFG.core_tickers)
        last_day = clean[clean["date"] == clean["date"].max()]
        latest_prices = {r.tic: float(r.close) for r in last_day.itertuples()}

        result = LI.decide(
            model_name=model_name, algo=algo, model_path=model_path, state_dir=cfg.STATE_DIR,
            clean_full_history=clean, scaling_stats=scaling_stats,
            current_equity=current_equity, current_position_values=current_position_values,
            latest_prices=latest_prices,
        )

        # ---- กรณี Dry-Run ----
        if not execute:
            print("(dry-run — ไม่ได้ส่ง order จริง)")
            if notify:
                text = TG.build_report(bridge, bridge.creds, scheme, algo, executed=False, result=result)
                print(text)
                TG.send(text)
            return result

        # ---- กรณีสั่ง --execute (ยิง Order จริง) ----
        for o in result.orders:
            client_order_id = f"{model_name}_{o.ticker}"[:40]
            try:
                resp = bridge.submit_market_on_open_order(
                    client_order_id=client_order_id, symbol=o.ticker, side=o.side, quantity=o.approx_qty,
                )
                print(f">> ส่ง {o.side} {o.approx_qty} {o.ticker} -> {resp}")
            except Exception as e:
                if "NON_TRADING_HOURS" in str(e):
                    print(f"⚠️ Webull Reject Order (วันหยุดนักขัตฤกษ์/นอกเวลาทำการ)")
                    if notify:
                        short_text = TG.build_closed_report(scheme, algo, "⚠️ ตลาดปิดทำการ (Webull Reject Order)")
                        print(short_text)
                        TG.send(short_text)
                    return result
                else:
                    raise e

        # ---- บันทึก CSV (เฉพาะตอนส่งจริง และตลาดเปิด) ----
        new_rows = None
        if result.orders:
            try:
                order_history = bridge.get_order_history()
                new_rows = TL.log_order_history_to_csv(scheme, algo, order_history)
                print(f">> บันทึก {len(new_rows)} order ใหม่ลง {TL.csv_path(scheme, algo)}")
            except Exception as e:
                print(f"[csv] บันทึกไม่สำเร็จ: {e}")

        # ---- แจ้งเตือน Telegram (รายงานฉบับเต็มเมื่อส่งสำเร็จ) ----
        if notify:
            text = TG.build_report(
                bridge, bridge.creds, scheme, algo, 
                executed=True, result=result, new_csv_rows=new_rows
            )
            print(text)
            TG.send(text)

        return result

    except Exception as e:
        # ---- 2. ดักจับ Error ทางเทคนิค / โค้ดพัง ทุกกรณี ----
        error_details = f"{type(e).__name__}: {str(e)}"
        print(f"\n🚨 เกิดข้อผิดพลาดในการรัน {scheme}_{algo}: {error_details}")
        traceback.print_exc()

        if notify:
            error_text = TG.build_error_report(scheme, algo, error_details[:200])
            TG.send(error_text)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["sliding_window", "expanding_window", "single_holdout"])
    ap.add_argument("--algo", required=True, choices=["ppo", "a2c", "sac", "td3"])
    ap.add_argument("--fetch-end", default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    fetch_end = args.fetch_end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    execute = args.execute and not args.dry_run
    run_one(args.scheme, args.algo, execute, fetch_end, notify=not args.no_notify)


if __name__ == "__main__":
    main()