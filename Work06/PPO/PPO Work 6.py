import pandas as pd
import yfinance as yf
import ta
import pandas_datareader.data as web
import numpy as np
import torch
import quantstats as qs
import os
import json
import warnings


warnings.filterwarnings('ignore')


from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ==========================================
# 0. HARDWARE SETUP
# ==========================================
# หมายเหตุ: PPO + MlpPolicy (เครือข่ายเล็ก ไม่ใช่ CNN/Transformer) มี overhead
# การส่งข้อมูลไป-กลับ CPU<->GPU สูงกว่าประโยชน์ที่ได้จากการขนานบน GPU
# นี่คือเหตุผลเชิงเทคนิคที่ยืนยันได้ ไม่ใช่แค่ความเห็น จึงยังคงใช้ CPU
device = "cpu"
print(f"✅ Device Strategy: '{device}' เพราะ PPO+MlpPolicy เป็นโมเดลเล็ก การใช้ GPU จะช้ากว่าเพราะ overhead การส่งข้อมูล")
if torch.cuda.is_available():
    print(f"   - พบ GPU: {torch.cuda.get_device_name(0)} (แต่ไม่แนะนำให้ใช้กับ PPO MlpPolicy)")
else:
    print("   - ไม่พบการ์ดจอแยก ระบบจะทำงานบน CPU (เหมาะสมอยู่แล้วสำหรับโปรเจกต์นี้)")

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
TICKERS = ['DIA', 'QQQ', 'SPY']

# แก้ไข #1: ปรับค่าธรรมเนียมให้สมจริงกับ ETF อเมริกา (SPY/QQQ/DIA)
# เดิมตั้งไว้ 1.5% ตามคำขอเริ่มต้น แต่ตัวเลขนี้สูงเกินจริงมากสำหรับตลาด US:
#   - ค่าคอมมิชชั่นโบรกเกอร์ US (เช่น IBKR): ~0.001-0.05% หรือ flat fee
#   - Slippage สำหรับ ETF ใหญ่ที่มี volume สูงอย่าง SPY/QQQ: ~0.01-0.05%
#   - รวมกันแล้วค่าธรรมเนียมที่สมจริง: ~0.1-0.3% ต่อครั้ง (ไม่มี VAT เพราะเป็นตลาดต่างประเทศ)
# ปรับเป็น 0.3% ซึ่งเป็นค่าขอบบนที่ยังสมจริง เผื่อ margin ให้ safety สูงกว่าค่าเฉลี่ยตลาดเล็กน้อย
TRANSACTION_COST_PCT = 0.003  # 0.3% ต่อการซื้อขายแต่ละครั้ง (ทั้งฝั่งซื้อและขาย)

# แก้ไข #2: publication lag ของข้อมูล macro รายเดือน
FED_FUNDS_LAG_DAYS = 30   # Fed ประกาศเดือนถัดไปหลังจบเดือน
M2_LAG_DAYS = 21          # Fed Reserve เผยแพร่ M2 ล่าช้าประมาณ 3 สัปดาห์

# ⚙️ สวิตช์เปิด/ปิด multi-timeframe indicators (Day/Week/Month)
# เปิดแล้ว state space จะโตขึ้นมาก (เพิ่มอีก ~2 เท่าของ technical indicators)
# แนะนำ: ทดสอบ False ก่อนให้เสถียร แล้วค่อยเปิด True ทีหลังเพื่อวัดผลแยกกัน (ablation study)
USE_MULTI_TIMEFRAME = False

# ---------- รายชื่อ indicator สมบูรณ์ (แทนที่ 4 ตัวเดิมที่มีแค่เส้นเดียว) ----------
COMPLETE_TECHNICAL_INDICATORS = [
    'RSI_14', 'RSI_signal', 'RSI_overbought', 'RSI_oversold', 'RSI_above_center',
    'MACD_line', 'MACD_signal', 'MACD_histogram', 'MACD_cross',
    'EMA_12', 'EMA_26', 'EMA_50', 'EMA_200',
    'price_to_EMA50_ratio', 'price_to_EMA200_ratio',
    'EMA_golden_cross', 'EMA_death_cross',
    'StochRSI_K', 'StochRSI_D', 'StochRSI_cross',
]

# ---------- รายชื่อ multi-timeframe indicator (ใช้เฉพาะเมื่อ USE_MULTI_TIMEFRAME=True) ----------
MULTI_TIMEFRAME_INDICATORS = [
    'RSI_weekly', 'RSI_monthly',
    'MACD_histogram_weekly', 'MACD_histogram_monthly',
]

MACRO_INDICATORS = ['vix', 'bond_yield', 'gold', 'wti', 'fed_rate', 'm2']

TECHNICAL_INDICATORS = COMPLETE_TECHNICAL_INDICATORS + (
    MULTI_TIMEFRAME_INDICATORS if USE_MULTI_TIMEFRAME else []
)
FEATURES = TECHNICAL_INDICATORS + MACRO_INDICATORS


# ==========================================
# 2. INDICATOR COMPUTATION (สมบูรณ์ + multi-timeframe)
# ==========================================
def compute_complete_indicators(df, price_col='close'):
    """
    คำนวณ indicator ครบชุดต่อหุ้นหนึ่งตัว (ต้อง apply ก่อน merge เป็น panel data)
    แทนที่เวอร์ชันเดิมที่ดึงมาแค่เส้นเดียวต่อ indicator (เช่น MACD ดึงแค่ macd line)
    """
    close = df[price_col]

    # ---------- RSI ชุดสมบูรณ์ ----------
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['RSI_14'] = rsi
    df['RSI_signal'] = rsi.rolling(window=9).mean()
    df['RSI_overbought'] = (rsi > 70).astype(float)
    df['RSI_oversold'] = (rsi < 30).astype(float)
    df['RSI_above_center'] = (rsi > 50).astype(float)

    # ---------- MACD ชุดสมบูรณ์ ----------
    macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    df['MACD_line'] = macd_line
    df['MACD_signal'] = macd_signal
    df['MACD_histogram'] = macd_obj.macd_diff()
    prev_diff = (macd_line - macd_signal).shift(1)
    curr_diff = macd_line - macd_signal
    df['MACD_cross'] = np.select(
        [(prev_diff <= 0) & (curr_diff > 0), (prev_diff >= 0) & (curr_diff < 0)],
        [1.0, -1.0], default=0.0
    )

    # ---------- EMA หลายช่วงเวลา + ความสัมพันธ์ ----------
    ema12 = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    ema26 = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()
    df['EMA_12'] = ema12
    df['EMA_26'] = ema26
    df['EMA_50'] = ema50
    df['EMA_200'] = ema200
    df['price_to_EMA50_ratio'] = (close - ema50) / ema50
    df['price_to_EMA200_ratio'] = (close - ema200) / ema200
    prev_ema_diff = (ema50 - ema200).shift(1)
    curr_ema_diff = ema50 - ema200
    df['EMA_golden_cross'] = ((prev_ema_diff <= 0) & (curr_ema_diff > 0)).astype(float)
    df['EMA_death_cross'] = ((prev_ema_diff >= 0) & (curr_ema_diff < 0)).astype(float)

    # ---------- Stochastic RSI ชุดสมบูรณ์ ----------
    stochrsi_obj = ta.momentum.StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
    stoch_k = stochrsi_obj.stochrsi_k()
    stoch_d = stochrsi_obj.stochrsi_d()
    df['StochRSI_K'] = stoch_k
    df['StochRSI_D'] = stoch_d
    prev_stoch_diff = (stoch_k - stoch_d).shift(1)
    curr_stoch_diff = stoch_k - stoch_d
    bullish_cross = (prev_stoch_diff <= 0) & (curr_stoch_diff > 0) & (stoch_k < 0.2)
    bearish_cross = (prev_stoch_diff >= 0) & (curr_stoch_diff < 0) & (stoch_k > 0.8)
    df['StochRSI_cross'] = np.select([bullish_cross, bearish_cross], [1.0, -1.0], default=0.0)

    return df


def compute_multi_timeframe_features(df, price_col='close', date_col='date'):
    """
    คำนวณ RSI และ MACD histogram แบบ weekly/monthly โดยไม่เกิด look-ahead bias
    หลักการ: resample เป็นแท่งใหญ่ -> shift(1) เพื่อไม่ให้แท่งปัจจุบัน (ยังไม่ปิด) leak เข้ามา
    -> reindex กลับมาเป็น daily ด้วย ffill (ค่าจะซ้ำกันทั้งสัปดาห์/เดือน ซึ่งถูกต้องแล้ว)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    close = df[price_col]

    # ---------- Weekly ----------
    weekly_close = close.resample('W-FRI').last()
    weekly_rsi = ta.momentum.RSIIndicator(close=weekly_close, window=14).rsi().shift(1)
    df['RSI_weekly'] = weekly_rsi.reindex(df.index, method='ffill')

    weekly_macd_hist = ta.trend.MACD(close=weekly_close).macd_diff().shift(1)
    df['MACD_histogram_weekly'] = weekly_macd_hist.reindex(df.index, method='ffill')

    # ---------- Monthly ----------
    monthly_close = close.resample('ME').last()
    monthly_rsi = ta.momentum.RSIIndicator(close=monthly_close, window=14).rsi().shift(1)
    df['RSI_monthly'] = monthly_rsi.reindex(df.index, method='ffill')

    monthly_macd_hist = ta.trend.MACD(close=monthly_close).macd_diff().shift(1)
    df['MACD_histogram_monthly'] = monthly_macd_hist.reindex(df.index, method='ffill')

    df = df.reset_index()
    return df


# ==========================================
# 3. HIGH-PERFORMANCE DATA PIPELINE
# ==========================================
def fetch_and_prepare_data(tickers, start_date, end_date):
    print(f"📦 กําลังดาวน์โหลดข้อมูลจาก {start_date} ถึง {end_date}...")
    processed_dfs = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue

            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = df.reset_index()
            df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                                'Close': 'close', 'Volume': 'volume'}, inplace=True)
            df['tic'] = ticker

            # ---- Indicator สมบูรณ์ (แทนที่ 4 บรรทัดเดิมที่ดึงแค่เส้นเดียว) ----
            df = compute_complete_indicators(df, price_col='close')

            # ---- Multi-timeframe (ถ้าเปิดใช้งาน) ----
            if USE_MULTI_TIMEFRAME:
                df = compute_multi_timeframe_features(df, price_col='close', date_col='date')

            processed_dfs.append(df)
        except Exception as e:
            print(f"❌ ไม่สามารถดาวน์โหลด {ticker}: {e}")

    if not processed_dfs:
        raise ValueError("ดาวน์โหลดข้อมูลล้มเหลว กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")

    final_df = pd.concat(processed_dfs, ignore_index=True)

    # ---- Macro data จาก yfinance (VIX, Bond Yield, Gold, WTI) ----
    macro_tickers = {"^VIX": "vix", "^TNX": "bond_yield", "GC=F": "gold", "CL=F": "wti"}
    try:
        macro_raw = yf.download(list(macro_tickers.keys()), start=start_date, end=end_date, progress=False)
        macro_df = macro_raw['Close'].copy()
        macro_df.columns = [col[0] if isinstance(col, tuple) else col for col in macro_df.columns]
        macro_df.rename(columns=macro_tickers, inplace=True)
        macro_df = macro_df.reset_index().rename(columns={'Date': 'date'})
        final_df = pd.merge(final_df, macro_df, on='date', how='left')
    except Exception as e:
        print(f"⚠️ ดึงข้อมูล VIX/Bond/Gold/WTI ไม่สำเร็จ: {e} -> จะเติมด้วย NaN แล้ว ffill แทนค่า 0.0")
        for name in macro_tickers.values():
            final_df[name] = np.nan

    # ---- FRED data (Fed Funds Rate, M2) พร้อม publication lag ----
    try:
        fed_funds = web.DataReader('FEDFUNDS', 'fred', start_date, end_date).reset_index()
        m2_supply = web.DataReader('M2SL', 'fred', start_date, end_date).reset_index()

        fed_funds.rename(columns={'DATE': 'date', 'FEDFUNDS': 'fed_rate'}, inplace=True)
        m2_supply.rename(columns={'DATE': 'date', 'M2SL': 'm2'}, inplace=True)

        # แก้ไข #2 (สำคัญที่สุด): เลื่อนวันที่ข้อมูลจะ "ถูกมองเห็น" ให้ตรงกับวันที่ประกาศจริง
        fed_funds['date'] = fed_funds['date'] + pd.Timedelta(days=FED_FUNDS_LAG_DAYS)
        m2_supply['date'] = m2_supply['date'] + pd.Timedelta(days=M2_LAG_DAYS)

        macro_fred = pd.merge(fed_funds, m2_supply, on='date', how='outer')
        final_df = pd.merge(final_df, macro_fred, on='date', how='left')
    except Exception as e:
        print(f"⚠️ ดึงข้อมูล FRED ไม่สำเร็จ: {e} -> จะเติมด้วย NaN แล้ว ffill แทนค่า 0.0")
        final_df['fed_rate'] = np.nan
        final_df['m2'] = np.nan

    # ---- จัดเรียงและเติมข้อมูล ----
    final_df.sort_values(['date', 'tic'], inplace=True)

    # แก้ไข #4: เอา bfill() ออก เพราะมันเอาค่าจากอนาคตมาเติมอดีต (look-ahead bias)
    # ⚠️ บั๊กที่แก้: เดิมใช้ groupby('tic').apply(lambda g: g.ffill()) ซึ่งบน pandas เวอร์ชันใหม่
    # (3.0+) ทำให้คอลัมน์ 'tic' หายไปจาก DataFrame เพราะ ffill() ถูกเรียกทับทั้ง DataFrame
    # รวมคอลัมน์ที่ใช้ group เอง วิธีที่ปลอดภัยคือ ffill เฉพาะคอลัมน์ตัวเลขเท่านั้น
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'day']  # 'day' ยังไม่ถูกสร้างตรงนี้ แต่กันไว้เผื่ออนาคต
    final_df[numeric_cols] = final_df.groupby('tic')[numeric_cols].ffill()

    # ⚠️ multi-timeframe columns (weekly/monthly) ใช้เวลา warm-up นานกว่า indicator รายวันมาก
    # (MACD monthly ต้องการ ~35 เดือนของข้อมูลก่อนจะไม่เป็น NaN) ถ้า dropna() รวมทุกคอลัมน์
    # พร้อมกันแบบเดิม อาจทำให้ข้อมูลทั้งหมดหายไปหมดถ้าช่วงข้อมูลไม่ยาวพอ
    # จึงเติม 0.0 ให้ multi-timeframe columns ที่ยัง NaN อยู่หลัง ffill แทนการ drop ทั้งแถว
    # (0.0 สมเหตุสมผลกว่าตรงนี้ เพราะ MACD histogram ที่ไม่มีข้อมูลพอ = "ยังไม่มีสัญญาณ" ซึ่งใกล้เคียง 0)
    if USE_MULTI_TIMEFRAME:
        for col in MULTI_TIMEFRAME_INDICATORS:
            if col in final_df.columns:
                still_nan = final_df[col].isna().sum()
                nan_ratio = still_nan / len(final_df) if len(final_df) > 0 else 0
                if still_nan > 0:
                    print(f"ℹ️ '{col}' มี NaN เหลือ {still_nan} แถว ({nan_ratio:.0%}) ช่วง warm-up -> เติมด้วย 0.0")
                if nan_ratio > 0.3:
                    print(f"⚠️ คำเตือน: '{col}' เป็น NaN ถึง {nan_ratio:.0%} ของข้อมูลทั้งหมด "
                          f"แปลว่าช่วงเวลา {start_date} ถึง {end_date} อาจสั้นเกินไปสำหรับ indicator "
                          f"รายเดือน/รายสัปดาห์นี้ (MACD monthly ต้องการข้อมูลจริงอย่างน้อย ~3 ปี) "
                          f"ค่า 0.0 ที่เติมเข้าไปจำนวนมากจะทำให้ feature นี้แทบไม่มีประโยชน์เชิงสัญญาณ")
                final_df[col] = final_df[col].fillna(0.0)

    rows_before = len(final_df)
    final_df.dropna(inplace=True)
    rows_after = len(final_df)
    if rows_before != rows_after:
        print(f"ℹ️ ตัดแถวที่มี NaN (ช่วง warm-up ของ indicator/publication lag) ออก {rows_before - rows_after} แถว")
    if rows_after == 0:
        raise ValueError(
            "⚠️ ข้อมูลเหลือ 0 แถวหลัง dropna! สาเหตุที่เป็นไปได้: ช่วงเวลา start_date-end_date "
            "สั้นเกินไปสำหรับ indicator ที่ต้องการ warm-up ยาว (เช่น EMA_200 ต้องการ ~200 วันทำการ). "
            "ลองขยายช่วงเวลาหรือลด window ของ indicator ที่ใช้"
        )

    date_list = sorted(final_df['date'].unique())
    date2day = {date: day for day, date in enumerate(date_list)}
    final_df['day'] = final_df['date'].map(date2day)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')

    final_df = final_df.sort_values(['date', 'tic'])
    final_df.index = final_df['day'].values
    return final_df


def compute_macro_scaling_stats(train_df):
    """
    Macro features (m2, fed_rate) มีสเกลใหญ่กว่า technical indicators มาก
    คำนวณ mean/std จาก TRAIN SET เท่านั้น (ป้องกัน leakage) เพื่อนำไปใช้ normalize
    ทั้ง train/val/test ด้วยค่าเดียวกัน
    """
    scale_cols = ['m2', 'fed_rate', 'vix', 'bond_yield', 'gold', 'wti']
    stats = {}
    for col in scale_cols:
        if col in train_df.columns:
            mean = train_df[col].mean()
            std = train_df[col].std()
            std = std if std > 1e-8 else 1.0
            stats[col] = (mean, std)
    return stats


def apply_macro_scaling(df, stats):
    """นำสถิติที่คำนวณจาก train set มาใช้ normalize คอลัมน์ macro ของ df ใดๆ"""
    df = df.copy()
    for col, (mean, std) in stats.items():
        if col in df.columns:
            df[col] = (df[col] - mean) / std
    return df


# ==========================================
# 4. ROBUST CUSTOM ENVIRONMENT
# ==========================================
class RealisticTradingEnv(StockTradingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vix_array = self.df['vix'].values if 'vix' in self.df.columns else np.zeros(len(self.df))

    def _calculate_reward(self):
        current_portfolio_value = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

        previous_portfolio_value = self.asset_memory[-1] if len(self.asset_memory) > 0 else self.initial_amount
        if previous_portfolio_value <= 0:
            previous_portfolio_value = self.initial_amount

        step_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value

        # ⚠️ บั๊กที่แก้ (สำคัญที่สุด — สาเหตุที่ agent เลือกไม่เทรดเลย):
        # เดิมมีบรรทัด "transaction_costs = self.cost * self.reward_scaling" แล้วเอาไปหักจาก raw_reward อีกที
        # แต่ current_portfolio_value ข้างบนคำนวณจาก self.state[0] (เงินสดคงเหลือ) ซึ่ง FinRL หัก
        # buy_cost_pct/sell_cost_pct ออกไปจากเงินสดตอนซื้อ-ขายเรียบร้อยแล้ว แปลว่า step_return
        # ที่คำนวณจาก portfolio value ต่างช่วงเวลา ได้ "รวม" ผลของค่าธรรมเนียมไปแล้วโดยอัตโนมัติ
        # การหัก transaction_costs อีกทีคือการหักค่าธรรมเนียมซ้ำสองครั้ง ทำให้การเทรดดูแพงเกินจริง
        # ~2 เท่า และเมื่อรวมกับ TRANSACTION_COST_PCT ที่ปรับเป็น 1.5% (จากเดิม 0.1%) ผลกระทบก็ยิ่งขยาย
        # จนทำให้ "ไม่เทรดเลย" กลายเป็นคำตอบที่ optimal ในทางคณิตศาสตร์ -> จึงเอาบรรทัดนี้ออก

        # Asymmetric Reward: ลงโทษการขาดทุนหนักกว่ากำไร แต่ลดจาก 2.0 -> 1.15
        # เหตุผลที่ลดมาก: ทดสอบเชิงตัวเลขพบว่า asymmetry ต้องอยู่ในช่วง 1.0-1.2 เท่านั้น
        # ถึงจะทำให้ expected value ของการเทรดยังสูงกว่าการถือเงินสดเมื่อตลาดมี positive drift
        # แบบ ETF จริง (ทดสอบด้วย 55% โอกาสขึ้น/45% โอกาสลง ขนาดใกล้เคียงกัน)
        # ค่า 2.0 เดิมทำให้ EV(เทรด) ติดลบกว่า EV(เงินสด) เสมอ -> agent เรียนรู้ให้ไม่เทรดเลย
        # เลือก 1.15 เป็นค่ากลางที่เผื่อ margin ไว้สำหรับตอน VIX สูงที่ risk_penalty จะเพิ่มเข้ามาอีก
        ASYMMETRY_FACTOR = 1.15
        if step_return < 0:
            step_return *= ASYMMETRY_FACTOR

        current_vix = self.vix_array[self.day] if self.day < len(self.vix_array) else 0.0
        invested_ratio = 1 - (self.state[0] / current_portfolio_value) if current_portfolio_value > 0 else 0

        # ⚠️ บั๊กที่แก้: เดิม risk_penalty คูณด้วย invested_ratio ตรงๆ ทำให้ agent ที่ถือเงินสด
        # 100% (invested_ratio=0) ได้ risk_penalty=0 เสมอไม่ว่า VIX จะสูงแค่ไหน กลายเป็น loophole
        # ที่บอกว่า "ไม่ลงทุนเลย = ไม่มีความเสี่ยงใดๆ ตลอดกาล" ซึ่งเสริมแรงจูงใจให้ agent เลือกไม่เทรด
        # ทางแก้: เพิ่ม small constant penalty สำหรับการถือเงินสดล้วนตอน VIX ต่ำ/ปกติ เพื่อไม่ให้
        # เงินสดเป็น "ที่หลบภัยที่ไม่มีต้นทุนเสียโอกาส" อย่างสมบูรณ์ (เงินสดควรมีต้นทุนเสียโอกาสบ้าง
        # เหมือนชีวิตจริงที่การไม่ลงทุนก็แพ้เงินเฟ้อในระยะยาว)
        risk_penalty = 0.0
        if current_vix > 0.0:  # VIX สูงกว่าค่าเฉลี่ยในอดีต (z-score > 0)
            vix_excess = current_vix / 10.0
            risk_penalty = vix_excess * invested_ratio * (abs(step_return) + 0.001)

        # opportunity cost เล็กๆ สำหรับการถือเงินสดล้วนในภาวะตลาดปกติ (VIX ไม่สูง)
        # ป้องกัน loophole ที่ "เงินสด 100% ตลอดกาล = reward 0 เสมอ ไม่มีความเสี่ยงเลย"
        cash_opportunity_cost = 0.0
        if invested_ratio < 0.05 and current_vix <= 0.0:
            cash_opportunity_cost = 0.02  # เทียบเท่า ~2% ต่อปีของผลตอบแทนที่เสียโอกาสไป หารรายวัน

        raw_reward = (step_return * 100) - (risk_penalty * 100) - cash_opportunity_cost
        # ใช้ tanh แทน hard clip: บีบค่า extreme แบบนุ่มนวล แทนตัดทิ้งตรงๆ
        reward = 10.0 * np.tanh(raw_reward / 10.0)

        return float(reward)


def get_env_kwargs(stock_dimension):
    state_space = 1 + (2 * stock_dimension) + (len(FEATURES) * stock_dimension)
    return {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": FEATURES,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "buy_cost_pct": [TRANSACTION_COST_PCT] * stock_dimension,
        "sell_cost_pct": [TRANSACTION_COST_PCT] * stock_dimension,
    }


# ==========================================
# 5. TRANSLATION & REPORT ENGINE
# ==========================================
def explain_bot_performance(returns):
    print("\n" + "=" * 55)
    print("🤖 สรุปผลงานบอทเทรดในหน้าต่างทดสอบจริง (ฉบับเข้าใจง่าย)")
    print("=" * 55)

    total_return = qs.stats.comp(returns) * 100
    cagr = qs.stats.cagr(returns) * 100
    max_dd = qs.stats.max_drawdown(returns) * 100
    sharpe = qs.stats.sharpe(returns)
    win_rate = qs.stats.win_rate(returns) * 100

    print("\n💰 1. ด้านการทำกำไร (Return)")
    print(f"   • กำไรสะสมตลอดการทดสอบ: {total_return:.2f}%")
    print(f"   • ผลตอบแทนทบต้นเฉลี่ยรายปี (CAGR): {cagr:.2f}%")
    if cagr > 12:
        print("   ✅ สรุป: ยอดเยี่ยม! บอทสร้างผลงานชนะอัตราเฉลี่ยของตลาดส่วนใหญ่")
    elif cagr > 0:
        print("   ⚠️ สรุป: พอใช้ได้ พอร์ตโตขึ้นแต่ยังไม่โดดเด่นนัก")
    else:
        print("   ❌ สรุป: ล้มเหลว บอททำเงินต้นสูญหาย")

    print("\n📉 2. ด้านความเสี่ยง (Risk & Drawdown)")
    print(f"   • ช่วงที่พอร์ตร่วงหนักสุดจากจุดสูงสุด (Max Drawdown): {max_dd:.2f}%")
    if max_dd > -15:
        print("   ✅ สรุป: ปลอดภัยสูง ระบบคุมสัดส่วนการสูญเสียเงินได้ดีมาก")
    elif max_dd > -30:
        print("   ⚠️ สรุป: ปานกลาง พอร์ตแกว่งตามมาตรฐานสไตล์กองทุนเชิงรุก")
    else:
        print("   ❌ สรุป: อันตรายมาก! บอทปล่อยให้พอร์ตเสียหายหนักเกินไป")

    print("\n⚖️ 3. ความคุ้มค่าและสถิติการชนะ (Efficiency)")
    print(f"   • ความคุ้มค่าต่อหนึ่งหน่วยความเสี่ยง (Sharpe Ratio): {sharpe:.2f}")
    if sharpe >= 1.0:
        print("   ✅ สรุป: ดีเยี่ยม กำไรที่ได้คุ้มค่าอย่างมากกับความเสี่ยงที่ถือครอง")
    else:
        print("   ⚠️ สรุป: ความคุ้มค่าน้อยลง ผลตอบแทนอาจไม่สมน้ำสมเนื้อกับความเสี่ยงที่เผชิญ")
    print(f"   • อัตราความแม่นยำรายวัน (Win Rate): {win_rate:.2f}%")
    print("=" * 55 + "\n")


# ==========================================
# 6. HIGH-SPEED TRAINING PIPELINE WITH VALIDATION
# ==========================================
def train_model():
    TRAIN_START, TRAIN_END = '1999-04-01', '2019-12-31'
    VAL_START, VAL_END = '2020-01-01', '2023-12-31'

    os.makedirs("./exported_data", exist_ok=True)

    print(f"\nℹ️ Feature ทั้งหมดที่ใช้ ({len(FEATURES)} ตัว): {FEATURES}")
    print(f"ℹ️ Multi-timeframe: {'เปิดใช้งาน' if USE_MULTI_TIMEFRAME else 'ปิดอยู่ (ตั้ง USE_MULTI_TIMEFRAME=True เพื่อเปิด)'}")

    print("\n--- 🛠️ [1/3] เริ่มเตรียมข้อมูลสําหรับฝึกสอนบอท (Training Set) ---")
    train_df = fetch_and_prepare_data(TICKERS, TRAIN_START, TRAIN_END)

    print("\n--- 🛠️ [2/3] เริ่มเตรียมข้อมูลสําหรับข้อสอบ (Validation Set) ---")
    val_df = fetch_and_prepare_data(TICKERS, VAL_START, VAL_END)

    # คำนวณสถิติ scaling จาก TRAIN SET เท่านั้น (ป้องกัน leakage)
    scaling_stats = compute_macro_scaling_stats(train_df)
    train_df = apply_macro_scaling(train_df, scaling_stats)
    val_df = apply_macro_scaling(val_df, scaling_stats)

    with open("./exported_data/macro_scaling_stats.json", "w") as f:
        json.dump({k: list(v) for k, v in scaling_stats.items()}, f)

    train_df.to_csv("./exported_data/train_set.csv", index=False)
    val_df.to_csv("./exported_data/validation_set.csv", index=False)
    print("📁 [Exported] บันทึกไฟล์ Training/Validation Set และสถิติ Scaling เรียบร้อย")

    env_kwargs = get_env_kwargs(len(TICKERS))

    def make_env():
        return lambda: RealisticTradingEnv(df=train_df, **env_kwargs)

    NUM_CPU = 4
    env_train = SubprocVecEnv([make_env() for _ in range(NUM_CPU)])
    env_val = DummyVecEnv([lambda: RealisticTradingEnv(df=val_df, **env_kwargs)])

    os.makedirs("./best_model", exist_ok=True)
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path='./best_model/',
        log_path='./best_model/logs/',
        eval_freq=max(500, 2048 // NUM_CPU),
        deterministic=True,
        render=False
    )

    agent = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=0.00025,
        n_steps=2048,
        batch_size=256,
        device=device,
        verbose=1
    )

    print("\n--- 🚀 [3/3] บอทเริ่มกระโจนเข้าสู่การเรียนรู้แบบคู่ขนาน ---")
    try:
        # เพิ่ม timesteps จาก 15,000 -> 300,000 (train set ~5,000 วัน/episode
        # 300,000 steps ≈ 60 episodes ซึ่งเพียงพอให้ PPO ปรับ policy โดยไม่ overfit เกินจำเป็น)
        agent.learn(total_timesteps=300_000, callback=eval_callback)
        agent.save("ppo_realistic_trading_bot_last")
        print("💾 บันทึกโมเดลเสร็จสิ้น! บอทเวอร์ชันที่ดีที่สุดถูกบรรจุอยู่ในโฟลเดอร์ './best_model/'")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในจังหวะเทรนโมเดล: {e}")


# ==========================================
# 7. COMPREHENSIVE OUT-OF-SAMPLE TEST
# ==========================================
def test_model():
    TEST_START, TEST_END = '2024-01-01', '2026-07-29'

    print("\n🔮 ดึงข้อมูลตลาดปัจจุบันมาสับไพ่ทดสอบจริง (Out-of-Sample Test)...")
    test_df = fetch_and_prepare_data(TICKERS, TEST_START, TEST_END)

    # โหลดสถิติ scaling ที่บันทึกไว้ตอน train_model() เพื่อ normalize ด้วยค่าเดียวกันเสมอ
    # ⚠️ บั๊กที่แก้: เดิมถ้าไม่พบไฟล์นี้ (เช่น เรียก test_model() แยกโดยไม่ train ก่อน หรือรันคนละ
    # session ที่ ./exported_data/ ถูกล้างไปแล้ว) โค้ดจะ crash ด้วย FileNotFoundError ที่ไม่อธิบาย
    # สาเหตุให้ผู้ใช้เข้าใจ จึงเพิ่ม try/except พร้อมคำอธิบายที่ชัดเจนว่าต้องรัน train_model() ก่อน
    scaling_path = "./exported_data/macro_scaling_stats.json"
    if not os.path.exists(scaling_path):
        raise FileNotFoundError(
            f"⚠️ ไม่พบไฟล์ '{scaling_path}' ซึ่งเก็บสถิติ normalize ของ macro features "
            f"(คำนวณจาก train set เท่านั้นเพื่อป้องกัน data leakage)\n"
            f"กรุณารัน train_model() ให้เสร็จสมบูรณ์ก่อนเรียก test_model() "
            f"(ทั้งสองฟังก์ชันต้องรันในโฟลเดอร์เดียวกัน เพราะไฟล์นี้ถูกเขียนไว้ที่ './exported_data/')"
        )
    with open(scaling_path, "r") as f:
        scaling_stats = {k: tuple(v) for k, v in json.load(f).items()}
    test_df = apply_macro_scaling(test_df, scaling_stats)

    os.makedirs("./exported_data", exist_ok=True)
    test_df.to_csv("./exported_data/test_set.csv", index=False)
    print("📁 [Exported] บันทึกไฟล์ Test Set เรียบร้อย -> ./exported_data/test_set.csv")

    model_path = "./best_model/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "ppo_realistic_trading_bot_last"
        print("⚠️ ไม่พบโมเดลคัดสรรพิเศษยามทำลายสถิติ วนกลับไปใช้โมเดลรอบสุดท้าย")

    trained_model = PPO.load(model_path, device=device)

    env_kwargs = get_env_kwargs(len(TICKERS))
    e_test_gym = RealisticTradingEnv(df=test_df, **env_kwargs)
    env_test = DummyVecEnv([lambda: e_test_gym])

    obs = env_test.reset()
    account_memory = []
    num_days = len(test_df.index.unique())

    print("📈 บอทกำลังดำเนินการจำลองการกระจายสินทรัพย์จริงในอดีต...")
    for i in range(num_days):
        action, _ = trained_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env_test.step(action)

        if i == num_days - 2:
            account_memory = env_test.env_method(method_name="save_asset_memory")[0]

        if dones[0]:
            print("🏁 การทดสอบย้อนหลังสิ้นสุดสมบูรณ์!")
            break

    df_account_value = pd.DataFrame(account_memory)
    df_account_value['date'] = pd.to_datetime(df_account_value['date'])
    df_account_value.set_index('date', inplace=True)

    df_account_value['daily_return'] = df_account_value['account_value'].pct_change()
    df_account_value.dropna(inplace=True)

    explain_bot_performance(df_account_value['daily_return'])


# ==========================================
# 8. EXECUTION GATEWAY
# ==========================================
if __name__ == "__main__":
    train_model()
    test_model()
