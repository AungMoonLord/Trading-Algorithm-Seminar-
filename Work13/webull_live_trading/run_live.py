"""
run_live.py
============
Entrypoint เดียวสำหรับรันเทรดจริง (paper) — เลือก scheme + algo

    python run_live.py --scheme sliding_window --algo ppo --dry-run
    python run_live.py --scheme sliding_window --algo ppo --execute

ต้องรัน fetch_scaling_stats.py --scheme <scheme> มาก่อนอย่างน้อย 1 ครั้ง และ
ต้องมีไฟล์ schemes/<scheme>/models/<algo>/best_model.zip อยู่แล้ว

--dry-run คือค่า default เสมอ ต้องใส่ --execute ชัดเจนถึงจะยิง order จริงเข้า Webull
"""
import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

import common as C
import live_inference as LI
import telegram_report as TG
import telegram_csv as TL
from webull_bridge import WebullBridge, WebullCredentials


def model_path_for(cfg, algo: str) -> Path:
    """โครงสร้างไฟล์: schemes/<scheme>/models/<algo>/fold_<N>/best_model.zip
    ใช้ cfg.ACTIVE_FOLD_ID เป็นตัวกำหนดว่าจะโหลด fold ไหนมา deploy จริง
    (fold อื่นๆ เก็บไว้เป็นของอ้างอิง/debug เท่านั้น ไม่ได้ถูกใช้ตอนรัน)"""
    return cfg.MODELS_DIR / algo / f"fold_{cfg.ACTIVE_FOLD_ID}" / "best_model.zip"


def parse_account_equity(acct: dict) -> float:
    """field จริงที่ยืนยันแล้วจาก response ของ Webull sandbox: total_net_liquidation_value"""
    if "total_net_liquidation_value" not in acct:
        raise KeyError(f"ไม่พบ total_net_liquidation_value ใน account summary: {acct}")
    return float(acct["total_net_liquidation_value"])


def parse_position_values(positions_raw, tickers) -> dict:
    """คืน {ticker: market_value} เฉพาะ 5 ตัวของเรา
    field name ยืนยันแล้วจาก response จริง: symbol, quantity, last_price, market_value"""
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
                raise KeyError(f"ไม่รู้จัก field ของ position นี้ (ส่งข้อความนี้ให้ผู้ช่วยดู): {it}")
            mv = float(qty) * float(px)
        out[sym] = float(mv)
    return out


def run_one(scheme: str, algo: str, execute: bool, fetch_end: str, bridge: WebullBridge = None,
            notify: bool = True):
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

    print(f"\n{'='*70}\n{model_name}  |  {cfg.NOTES}\n{'='*70}")
    scaling_stats = C.load_scaling_stats(cfg.SCALING_STATS_PATH)

    if bridge is None:
        creds = WebullCredentials.from_env()
        bridge = WebullBridge(creds)

    acct = bridge.get_account_summary()
    positions_raw = bridge.get_positions()
    print(">> Account summary (raw):", acct)
    print(">> Positions (raw):", positions_raw)

    raw = C.fetch_and_stitch_universe(C.CFG.core_tickers, "2024-01-01", fetch_end)
    feats = C.compute_institutional_features(raw, C.CFG.core_tickers)
    clean = C.clean_and_align_dataset(feats, C.CFG.core_tickers)

    current_equity = parse_account_equity(acct)
    current_position_values = parse_position_values(positions_raw, C.CFG.core_tickers)
    # ราคาล่าสุดใช้ราคาปิดวันล่าสุดจากข้อมูลที่ดึงมา (ใช้แค่ประมาณจำนวนหุ้นที่ต้องซื้อ/ขาย)
    last_day = clean[clean["date"] == clean["date"].max()]
    latest_prices = {r.tic: float(r.close) for r in last_day.itertuples()}
    print(f">> equity={current_equity:,.2f}  positions={current_position_values}")
    print(f">> ข้อมูลตลาดล่าสุด: {clean['date'].max().date()}  prices={latest_prices}")

    result = LI.decide(
        model_name=model_name, algo=algo, model_path=model_path, state_dir=cfg.STATE_DIR,
        clean_full_history=clean, scaling_stats=scaling_stats,
        current_equity=current_equity, current_position_values=current_position_values,
        latest_prices=latest_prices,
    )

    print("target_weights :", result.target_weights)
    print("current_weights:", result.current_weights)
    print(f"turnover       : {result.turnover:.4f}")
    print(f"risk scalar    : {result.scalar:.3f}  realized_vol: {result.realized_vol:.5f}")
    print("orders:")
    for o in result.orders:
        print(" ", o)

    if not execute:
        print("(dry-run — ไม่ได้ส่ง order จริง)")
        if notify:
            text = TG.build_report(bridge, bridge.creds, scheme, algo, executed=False, result=result)
            print(text)
            TG.send(text)
        return result

    for o in result.orders:
        client_order_id = f"{model_name}_{o.ticker}"[:40]
        resp = bridge.submit_market_on_open_order(
            client_order_id=client_order_id, symbol=o.ticker, side=o.side, quantity=o.approx_qty,
        )
        print(f">> ส่ง {o.side} {o.approx_qty} {o.ticker} -> {resp}")

    # ---- บันทึก CSV (เฉพาะตอนส่งจริง) ----
    new_rows = None
    if result.orders:
        try:
            order_history = bridge.get_order_history()
            new_rows = TL.log_order_history_to_csv(scheme, algo, order_history)
            print(f">> บันทึก {len(new_rows)} order ใหม่ลง {TL.csv_path(scheme, algo)}")
        except Exception as e:
            print(f"[csv] บันทึกไม่สำเร็จ (ไม่กระทบ order ที่ส่งไปแล้ว): {e}")

    # ---- แจ้งเตือน Telegram ----
    if notify:
        text = TG.build_report(bridge, bridge.creds, scheme, algo, executed=True,
                                result=result, new_csv_rows=new_rows)
        print(text)
        TG.send(text)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["sliding_window", "expanding_window", "single_holdout"])
    ap.add_argument("--algo", required=True, choices=["ppo", "a2c", "sac", "td3"])
    ap.add_argument("--fetch-end", default=None, help="วันที่สิ้นสุดดึงข้อมูล (default: พรุ่งนี้ เพราะ yfinance นับ end แบบไม่รวมวันนั้น)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                     help="ไม่ต้องใส่ก็ได้ (เป็นค่า default อยู่แล้ว) ใส่ไว้เผื่ออยากเขียนให้ชัดเจนในคำสั่ง — ไม่มีผลใดๆ แค่ทำให้ไม่ error ถ้าพิมพ์มา")
    ap.add_argument("--no-notify", action="store_true", help="ปิดการส่ง Telegram รอบนี้ (ยังบันทึก CSV ตามปกติถ้า --execute)")
    args = ap.parse_args()

    fetch_end = args.fetch_end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    execute = args.execute and not args.dry_run  # --dry-run ชนะเสมอ ถ้าเผลอใส่ทั้งคู่ (ปลอดภัยไว้ก่อน)
    run_one(args.scheme, args.algo, execute, fetch_end, notify=not args.no_notify)


if __name__ == "__main__":
    main()