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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# 📌 [แก้ตรงนี้: เปลี่ยนการอิมพอร์ตจาก PPO เป็น SAC]
from stable_baselines3 import SAC


# ==========================================
# 0. HARDWARE SETUP
# ==========================================
device = "cpu"
print(f"✅ Device Strategy: '{device}' เพราะ PPO+MlpPolicy เป็นโมเดลเล็ก การใช้ GPU จะช้ากว่าเพราะ overhead การส่งข้อมูล")
if torch.cuda.is_available():
    print(f"   - พบ GPU: {torch.cuda.get_device_name(0)} (แต่ไม่แนะนำให้ใช้กับ PPO MlpPolicy)")
else:
    print("   - ไม่พบการ์ดจอแยก ระบบจะทำงานบน CPU")


# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
TICKERS = ['DIA', 'QQQ', 'SPY']
TRANSACTION_COST_PCT = 0.003  

FED_FUNDS_LAG_DAYS = 30   
M2_LAG_DAYS = 21          

USE_MULTI_TIMEFRAME = True # แนะนำให้เปิดได้แล้ว เพราะแก้ปัญหาเรื่อง NaN Buffer แล้ว

# แก้ไข #1: เปลี่ยนโครงสร้าง Technical Indicators ให้เป็น Stationary ทั้งหมด (ไม่มีราคาดิบเจือปน)
# - ถอด EMA_12, 26, 50, 200 (ที่เป็นราคาดิบ) ออกทั้งหมด
# - ใช้สัดส่วน (Ratio) และเปอร์เซ็นต์ (Pct) แทน เพื่อให้ค่าแกว่งตัวในกรอบแคบและสเกลคงที่ไม่ว่าราคาหุ้นจะ 100 หรือ 500
COMPLETE_TECHNICAL_INDICATORS = [
    'RSI_14', 'RSI_signal', 'RSI_overbought', 'RSI_oversold', 'RSI_above_center',
    'MACD_line_pct', 'MACD_signal_pct', 'MACD_hist_pct', 'MACD_cross',
    'EMA_12_26_ratio', 'EMA_50_200_ratio', 'price_to_EMA50_ratio', 'price_to_EMA200_ratio',
    'EMA_golden_cross', 'EMA_death_cross',
    'StochRSI_K', 'StochRSI_D', 'StochRSI_cross',
]

MULTI_TIMEFRAME_INDICATORS = [
    'RSI_weekly', 'RSI_monthly',
    'MACD_hist_weekly_pct', 'MACD_hist_monthly_pct',
]

MACRO_INDICATORS = ['vix', 'bond_yield', 'gold', 'wti', 'fed_rate', 'm2']

TECHNICAL_INDICATORS = COMPLETE_TECHNICAL_INDICATORS + (
    MULTI_TIMEFRAME_INDICATORS if USE_MULTI_TIMEFRAME else []
)
FEATURES = TECHNICAL_INDICATORS + MACRO_INDICATORS


# ==========================================
# 2. INDICATOR COMPUTATION (Stationary & Scaled)
# ==========================================
def compute_complete_indicators(df, price_col='close'):
    close = df[price_col]

    # แก้ไข #2: Normalize RSI ให้ร่วงมาอยู่ในกรอบ [0, 1] เสมอ
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['RSI_14'] = rsi / 100.0 
    df['RSI_signal'] = rsi.rolling(window=9).mean() / 100.0
    df['RSI_overbought'] = (rsi > 70).astype(float)
    df['RSI_oversold'] = (rsi < 30).astype(float)
    df['RSI_above_center'] = (rsi > 50).astype(float)

    # แก้ไข #3: แปลง MACD เป็น PPOscillator (หารด้วย EMA26) ป้องกันโมเดลช็อกถ้าราคาพุ่งไป 500+
    ema26 = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()
    macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()

    df['MACD_line_pct'] = macd_line / ema26
    df['MACD_signal_pct'] = macd_signal / ema26
    df['MACD_hist_pct'] = macd_obj.macd_diff() / ema26

    prev_diff = (macd_line - macd_signal).shift(1)
    curr_diff = macd_line - macd_signal
    df['MACD_cross'] = np.select(
        [(prev_diff <= 0) & (curr_diff > 0), (prev_diff >= 0) & (curr_diff < 0)],
        [1.0, -1.0], default=0.0
    )

    # แก้ไข #4: เปลี่ยน EMA ดิบเป็น EMA Ratio (Stationary Feature) เพื่อไม่ให้ Gradient ระเบิด
    ema12 = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    df['EMA_12_26_ratio'] = (ema12 - ema26) / ema26
    df['EMA_50_200_ratio'] = (ema50 - ema200) / ema200
    df['price_to_EMA50_ratio'] = (close - ema50) / ema50
    df['price_to_EMA200_ratio'] = (close - ema200) / ema200

    prev_ema_diff = (ema50 - ema200).shift(1)
    curr_ema_diff = ema50 - ema200
    df['EMA_golden_cross'] = ((prev_ema_diff <= 0) & (curr_ema_diff > 0)).astype(float)
    df['EMA_death_cross'] = ((prev_ema_diff >= 0) & (curr_ema_diff < 0)).astype(float)

    # StochRSI มีกรอบ 0-1 ในตัวมันเองอยู่แล้ว ไม่ต้องหารเพิ่ม
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
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    close = df[price_col]

    weekly_close = close.resample('W-FRI').last()
    weekly_rsi = ta.momentum.RSIIndicator(close=weekly_close, window=14).rsi().shift(1)
    df['RSI_weekly'] = (weekly_rsi / 100.0).reindex(df.index, method='ffill')

    weekly_macd_hist = ta.trend.MACD(close=weekly_close).macd_diff().shift(1)
    # ทำ MACD ให้เป็น Percent (หารด้วยราคาก่อนหน้า)
    df['MACD_hist_weekly_pct'] = (weekly_macd_hist / weekly_close.shift(1)).reindex(df.index, method='ffill')

    monthly_close = close.resample('ME').last()
    monthly_rsi = ta.momentum.RSIIndicator(close=monthly_close, window=14).rsi().shift(1)
    df['RSI_monthly'] = (monthly_rsi / 100.0).reindex(df.index, method='ffill')

    monthly_macd_hist = ta.trend.MACD(close=monthly_close).macd_diff().shift(1)
    df['MACD_hist_monthly_pct'] = (monthly_macd_hist / monthly_close.shift(1)).reindex(df.index, method='ffill')

    df = df.reset_index()
    return df


# ==========================================
# 3. HIGH-PERFORMANCE DATA PIPELINE
# ==========================================
# แก้ไข #5: เพิ่มพารามิเตอร์ trim_start สำหรับการทำ Lookback Buffer
# เพื่อให้เราดึงข้อมูลย้อนหลังไปไกลๆ ได้ (warm-up indicator) แล้วค่อยตัดเฉพาะช่วง Test จริงทีหลัง
def fetch_and_prepare_data(tickers, start_date, end_date, trim_start=None):
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
            df = compute_complete_indicators(df, price_col='close')

            if USE_MULTI_TIMEFRAME:
                df = compute_multi_timeframe_features(df, price_col='close', date_col='date')

            processed_dfs.append(df)
        except Exception as e:
            print(f"❌ ไม่สามารถดาวน์โหลด {ticker}: {e}")

    if not processed_dfs:
        raise ValueError("ดาวน์โหลดข้อมูลล้มเหลว กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")

    final_df = pd.concat(processed_dfs, ignore_index=True)

    macro_tickers = {"^VIX": "vix", "^TNX": "bond_yield", "GC=F": "gold", "CL=F": "wti"}
    try:
        macro_raw = yf.download(list(macro_tickers.keys()), start=start_date, end=end_date, progress=False)
        macro_df = macro_raw['Close'].copy()
        macro_df.columns = [col[0] if isinstance(col, tuple) else col for col in macro_df.columns]
        macro_df.rename(columns=macro_tickers, inplace=True)
        macro_df = macro_df.reset_index().rename(columns={'Date': 'date'})
        final_df = pd.merge(final_df, macro_df, on='date', how='left')
    except:
        for name in macro_tickers.values():
            final_df[name] = np.nan

    try:
        fed_funds = web.DataReader('FEDFUNDS', 'fred', start_date, end_date).reset_index()
        m2_supply = web.DataReader('M2SL', 'fred', start_date, end_date).reset_index()
        fed_funds.rename(columns={'DATE': 'date', 'FEDFUNDS': 'fed_rate'}, inplace=True)
        m2_supply.rename(columns={'DATE': 'date', 'M2SL': 'm2'}, inplace=True)

        fed_funds['date'] = fed_funds['date'] + pd.Timedelta(days=FED_FUNDS_LAG_DAYS)
        m2_supply['date'] = m2_supply['date'] + pd.Timedelta(days=M2_LAG_DAYS)

        macro_fred = pd.merge(fed_funds, m2_supply, on='date', how='outer')
        final_df = pd.merge(final_df, macro_fred, on='date', how='left')
    except:
        final_df['fed_rate'] = np.nan
        final_df['m2'] = np.nan

    final_df.sort_values(['date', 'tic'], inplace=True)

    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'day']
    final_df[numeric_cols] = final_df.groupby('tic')[numeric_cols].ffill()

    if USE_MULTI_TIMEFRAME:
        for col in MULTI_TIMEFRAME_INDICATORS:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0.0)

    final_df.dropna(inplace=True)

    # แก้ไข #6: ตัดข้อมูลเฉพาะช่วงทดสอบจริง (หลัง Warm-up เสร็จ) เพื่อลดปัญหา NaN 100% 
    if trim_start is not None:
        final_df = final_df[final_df['date'] >= pd.to_datetime(trim_start)]
        print(f"✂️ หั่น Lookback Buffer ทิ้ง เหลือข้อมูลใช้งานจริงตั้งแต่: {trim_start}")

    # รัน index วันที่ 'day' ใหม่หลังจากตัด Buffer ออก เพื่อให้ Env Index ไม่เพี้ยน
    date_list = sorted(final_df['date'].unique())
    date2day = {date: day for day, date in enumerate(date_list)}
    final_df['day'] = final_df['date'].map(date2day)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')

    final_df = final_df.sort_values(['date', 'tic'])
    final_df.index = final_df['day'].values
    return final_df


# แก้ไข #7: ปรับโครงสร้างระบบ Normalization ขยายการครอบคลุม
# นำเอา Continuous Features ทั้ง Macro และ Technical เข้าสู่กระบวนการ Z-Score
def compute_feature_scaling_stats(train_df):
    scale_cols = [
        # Macro
        'm2', 'fed_rate', 'vix', 'bond_yield', 'gold', 'wti',
        # Continuous Technical (Ratios & Percentages)
        'MACD_line_pct', 'MACD_signal_pct', 'MACD_hist_pct',
        'EMA_12_26_ratio', 'EMA_50_200_ratio', 
        'price_to_EMA50_ratio', 'price_to_EMA200_ratio'
    ]
    if USE_MULTI_TIMEFRAME:
        scale_cols.extend(['MACD_hist_weekly_pct', 'MACD_hist_monthly_pct'])

    stats = {}
    for col in scale_cols:
        if col in train_df.columns:
            mean = train_df[col].mean()
            std = train_df[col].std()
            std = std if std > 1e-8 else 1.0
            stats[col] = (mean, std)
    return stats


def apply_feature_scaling(df, stats):
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

        # [ใหม่ 1/3] สร้างตัวแปรจำจุดสูงสุดของพอร์ตตอนเริ่ม Environment
        self.peak_portfolio_value = self.initial_amount

    # [ใหม่ 2/3] ต้อง Override ฟังก์ชัน reset() เพื่อล้างค่า Peak กลับเป็นทุนเริ่มต้นทุกครั้งที่เริ่มรอบใหม่
    # [แก้ไข] เพิ่ม **kwargs เพื่อรับพารามิเตอร์ซ่อนเร้นจาก Stable-Baselines3
    def reset(self,**kwargs):
        # โยน **kwargs ส่งต่อไปให้คลาสแม่ (FinRL) จัดการ
        state = super().reset(**kwargs) # เรียกใช้ reset ของคลาสแม่ (FinRL)
        self.peak_portfolio_value = self.initial_amount # ล้างค่า High Water Mark
        return state

    def _calculate_reward(self):
        current_portfolio_value = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

        # [ใหม่ 3/3] อัปเดตจุดสูงสุดของพอร์ต (High Water Mark) แบบ Real-time
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        previous_portfolio_value = self.asset_memory[-1] if len(self.asset_memory) > 0 else self.initial_amount
        if previous_portfolio_value <= 0:
            previous_portfolio_value = self.initial_amount

        step_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value

        # Asymmetric Reward ขาลง (หักคะแนนหนักกว่าตอนได้กำไร 50%)

        ASYMMETRY_FACTOR = 1.5
        if step_return < 0:
            step_return *= ASYMMETRY_FACTOR

        current_vix = self.vix_array[self.day] if self.day < len(self.vix_array) else 0.0
        invested_ratio = 1 - (self.state[0] / current_portfolio_value) if current_portfolio_value > 0 else 0

        # Risk Penalty จาก VIX
        risk_penalty = 0.0 
        if current_vix > 0.0:  # Z-score > 0 (VIX สูงกว่าค่าเฉลี่ยใน Training Set)
            vix_excess = current_vix / 10.0
            risk_penalty = vix_excess * invested_ratio * (abs(step_return) + 0.001)

        # ค่าเสียโอกาสหากถือเงินสดตอนตลาดปกติ
        cash_opportunity_cost = 0.0
        if invested_ratio < 0.05 and current_vix <= 0.0:
            cash_opportunity_cost = 0.02 

        # คำนวณ Reward เบื้องต้น (สเกล * 100 เพื่อให้ตัวเลขอยู่ในช่วงประมาณ -5 ถึง +5)
        raw_reward = (step_return * 100) - (risk_penalty * 100) - cash_opportunity_cost


        # สมมติเราเก็บค่าจุดสูงสุดของพอร์ตไว้ (self.peak_portfolio_value)

        current_drawdown = (current_portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value

        # ---------------------------------------------------------
        # 🚨 ส่วนที่เพิ่ม: DRAWDOWN PENALTY (ลงโทษเมื่อพอร์ตร่วงจากจุดสูงสุด)
        # ---------------------------------------------------------

        current_drawdown = (current_portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value
        # ถ้า Drawdown เริ่มลึกกว่า -5% ให้หักคะแนนหนักขึ้นเรื่อยๆ แบบ Exponential
        if current_drawdown < -0.05:
            dd_penalty = (abs(current_drawdown) - 0.05) * 10 
            raw_reward -= dd_penalty
        # ---------------------------------------------------------

        # ใช้ tanh ตัดขอบ reward ให้อยู่ในช่วง -10 ถึง +10 (ป้องกัน Gradient ระเบิดเวลาตลาดพังหนัก)
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
    TRAIN_START, TRAIN_END = '1999-01-01', '2019-12-31'
    VAL_START, VAL_END = '2020-01-01', '2023-12-31'

    os.makedirs("./exported_data", exist_ok=True)

    print(f"\nℹ️ Feature ทั้งหมดที่ใช้ ({len(FEATURES)} ตัว): {FEATURES}")
    print("\n--- 🛠️ [1/3] เริ่มเตรียมข้อมูลสําหรับฝึกสอนบอท (Training Set) ---")
    train_df = fetch_and_prepare_data(TICKERS, TRAIN_START, TRAIN_END)

    print("\n--- 🛠️ [2/3] เริ่มเตรียมข้อมูลสําหรับข้อสอบ (Validation Set) ---")
    val_df = fetch_and_prepare_data(TICKERS, VAL_START, VAL_END)

    # Scale ทั้ง Macro และ Technical (อิงจาก Train set เพื่อป้องกัน leakage)
    scaling_stats = compute_feature_scaling_stats(train_df)
    train_df = apply_feature_scaling(train_df, scaling_stats)
    val_df = apply_feature_scaling(val_df, scaling_stats)

    with open("./exported_data/feature_scaling_stats.json", "w") as f:
        json.dump({k: list(v) for k, v in scaling_stats.items()}, f)

    train_df.to_csv("./exported_data/train_set.csv", index=False)
    val_df.to_csv("./exported_data/validation_set.csv", index=False)

    env_kwargs = get_env_kwargs(len(TICKERS))
    def make_env():
        return lambda: RealisticTradingEnv(df=train_df, **env_kwargs)

    # 📌 [หมายเหตุ: สำหรับ Windows ถ้าเกิดปัญหา KeyboardInterrupt/ค้าง ให้ตั้ง NUM_CPU = 1 และใช้ DummyVecEnv เพียวๆ]
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

    # 📌 [แก้ตรงนี้: เปลี่ยน agent เป็น SAC และนำพารามิเตอร์ n_steps ออก]
    agent = SAC("MlpPolicy", env_train, learning_rate=0.00025, batch_size=256, device=device, verbose=1)

    print("\n--- 🚀 [3/3] บอทเริ่มกระโจนเข้าสู่การเรียนรู้แบบคู่ขนาน ---")
    try:
        # คุณน่าจะเห็น Explained Variance ขยับสูงขึ้นอย่างชัดเจนจากการรันรอบนี้
        agent.learn(total_timesteps=150_000, callback=eval_callback) 
        # 📌 [แก้ตรงนี้: เปลี่ยนชื่อไฟล์ที่บันทึกให้สอดคล้องกับ SAC]
        agent.save("sac_realistic_trading_bot_last")
        print("💾 บันทึกโมเดลเสร็จสิ้น!")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในจังหวะเทรนโมเดล: {e}")


# ==========================================
# 7. COMPREHENSIVE OUT-OF-SAMPLE TEST
# ==========================================
def test_model():
    # แก้ไข #8: เผื่อระยะเวลา 4 ปี (2020) เป็น Lookback Buffer ให้ MACD Monthly มีข้อมูลพอคำนวณ
    # และเริ่มประเมินผล Test จริงในวันที่ '2024-01-01' โดยใช้พารามิเตอร์ trim_start
    FETCH_START, TEST_END = '2020-01-01', '2026-07-29'
    TRIM_START = '2024-01-01'

    print("\n🔮 ดึงข้อมูลตลาดปัจจุบันมาสับไพ่ทดสอบจริง (Out-of-Sample Test)...")
    test_df = fetch_and_prepare_data(TICKERS, FETCH_START, TEST_END, trim_start=TRIM_START)

    scaling_path = "./exported_data/feature_scaling_stats.json"
    if not os.path.exists(scaling_path):
        raise FileNotFoundError(f"⚠️ ไม่พบไฟล์ '{scaling_path}' กรุณารัน train_model() ก่อน")

    with open(scaling_path, "r") as f:
        scaling_stats = {k: tuple(v) for k, v in json.load(f).items()}
    test_df = apply_feature_scaling(test_df, scaling_stats)

    os.makedirs("./exported_data", exist_ok=True)
    test_df.to_csv("./exported_data/test_set.csv", index=False)

    # 📌 [แก้ตรงนี้: ปรับแก้ชื่อไฟล์ที่จะทำการโหลดให้เป็นเวอร์ชันของ SAC]
    model_path = "./best_model/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "sac_realistic_trading_bot_last"

    # 📌 [แก้ตรงนี้: ปรับแก้เป็น SAC.load เพื่อให้โหลดโมเดลกลับมาถูกคลาส]
    trained_model = SAC.load(model_path, device=device)
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