"""
fetch_scaling_stats.py
=======================
สร้าง scaling_stats.json ของ scheme ที่เลือก (อ่านวันที่ train จาก
schemes/<scheme>/config.py) — รันครั้งเดียวต่อ scheme ก่อนเริ่มเทรดจริง

    python fetch_scaling_stats.py --scheme sliding_window
    python fetch_scaling_stats.py --scheme expanding_window
    python fetch_scaling_stats.py --scheme single_holdout

ต้องรันบนเครื่องที่ต่อเน็ตได้จริง (yfinance + FRED)
"""
import argparse
import importlib

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["sliding_window", "expanding_window", "single_holdout"])
    ap.add_argument("--fetch-end", default="2026-09-26", help="วันที่ล่าสุดที่จะดึงข้อมูล (default: วันนี้)")
    args = ap.parse_args()

    cfg = importlib.import_module(f"schemes.{args.scheme}.config")
    cfg.STATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f">> Scheme: {cfg.SCHEME_NAME}")
    print(f">> {cfg.NOTES}")

    raw = C.fetch_and_stitch_universe(C.CFG.core_tickers, C.CFG.global_fetch_start, args.fetch_end)
    feats = C.compute_institutional_features(raw, C.CFG.core_tickers)
    clean = C.clean_and_align_dataset(feats, C.CFG.core_tickers)

    train_slice = clean[(clean["date"] >= cfg.TRAIN_START) & (clean["date"] <= cfg.TRAIN_END)].copy()
    if train_slice.empty:
        raise RuntimeError("train_slice ว่างเปล่า — เช็ควันที่ใน config.py / การเชื่อมต่อ yfinance")

    stats = C.compute_scaling_stats(train_slice)
    C.save_scaling_stats(stats, cfg.SCALING_STATS_PATH)
    print(f">> Saved -> {cfg.SCALING_STATS_PATH}")
    for k, (mean, std) in stats.items():
        print(f"  {k:22s} mean={mean:+.5f}  std={std:.5f}")

    clean.to_parquet(cfg.STATE_DIR / "clean_features_full_history.parquet", index=False)
    print(f">> Saved -> {cfg.STATE_DIR / 'clean_features_full_history.parquet'}")


if __name__ == "__main__":
    main()
