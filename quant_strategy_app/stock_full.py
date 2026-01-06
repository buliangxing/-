import akshare as ak
import pandas as pd
import numpy as np
import logging
import sys
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from requests.exceptions import Timeout, RequestException
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================== 基础配置（专业金融+移动端适配） ======================
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')
# 专业金融图表样式 + 移动端适配
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'font.family': 'sans-serif',
    'figure.max_open_warning': 0,
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc'
})

# 专业日志配置
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Streamlit核心配置（专业金融风格）
st.set_page_config(
    page_title="专业股票分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "专业级实时股票分析系统（机构版）"}
)

# ====================== 全局配置优化 ======================
# 12个热门板块
HOT_SECTORS = [
    '人工智能', '半导体', '新能源汽车', '光伏', '消费电子', '医药生物', 
    '军工', '金融', '白酒', '锂电池', '算力', '储能'
]
# 板块-股票映射表（扩充+精准）
STOCK_SECTOR_MAP = {
    # 人工智能
    '002230': '人工智能', '300476': '人工智能', '600560': '人工智能', '000977': '人工智能',
    # 半导体
    '603986': '半导体', '002049': '半导体', '300661': '半导体', '688981': '半导体',
    # 新能源汽车
    '300750': '新能源汽车', '002594': '新能源汽车', '601633': '新能源汽车', '002460': '新能源汽车',
    # 光伏
    '300274': '光伏', '601012': '光伏', '002129': '光伏', '688599': '光伏',
    # 消费电子
    '002475': '消费电子', '002384': '消费电子', '300782': '消费电子', '601138': '消费电子',
    # 医药生物
    '600276': '医药生物', '300760': '医药生物', '002007': '医药生物', '688180': '医药生物',
    # 军工
    '600893': '军工', '002190': '军工', '600391': '军工', '300775': '军工',
    # 金融
    '601318': '金融', '600036': '金融', '601689': '金融', '000001': '金融',
    # 白酒
    '600519': '白酒', '000858': '白酒', '000596': '白酒', '600809': '白酒',
    # 锂电池
    '300750': '锂电池', '002460': '锂电池', '300073': '锂电池', '603799': '锂电池',
    # 算力
    '603019': '算力', '000977': '算力', '600410': '算力', '300308': '算力',
    # 储能
    '300274': '储能', '600406': '储能', '002594': '储能', '300802': '储能'
}
CACHE_TTL = 5  # 缓存5秒
global_spot_cache = None
cache_update_time = None
# 金融分析阈值
ANALYSIS_THRESHOLD = {
    '赚钱效应': 0.5, 'rsi_overbuy': 70, 'rsi_oversell': 30,
    'kdj_overbuy': 80, 'kdj_oversell': 20, 'fund_flow_positive': 0
}

# ====================== 核心数据处理函数（修复关键报错） ======================
def get_column_name_fixed(df, target_cols):
    """自适应列名匹配"""
    df_cols = df.columns.tolist()
    for col in target_cols:
        if col in df_cols:
            return col
    return target_cols[0]

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_spot_data_cached():
    """全市场行情缓存（修复板块列不存在的问题）"""
    global global_spot_cache, cache_update_time
    current_time = time.time()
    
    if global_spot_cache is not None and (current_time - cache_update_time) < CACHE_TTL:
        return global_spot_cache
    
    try:
        df = ak.stock_zh_a_spot_em()
        # 修复点1：先创建空的"板块"列，避免KeyError
        df['板块'] = ''
        
        # 列名映射（兼容不同接口返回）
        col_mapping = {
            '代码': '代码', '名称': '名称', '最新价': '最新价', '涨跌幅': '涨跌幅',
            '成交量': '成交量', '成交额': '成交额'
        }
        # 只重命名存在的列
        new_cols = {old: new for old, new in col_mapping.items() if old in df.columns}
        df = df.rename(columns=new_cols)
        
        # 修复点2：优先用映射表填充板块，避免依赖接口返回的板块列
        # 先确保"代码"列存在且为字符串
        if '代码' in df.columns:
            df['代码'] = df['代码'].astype(str).str.zfill(6)  # 补全6位代码
            # 用映射表填充板块
            df['板块'] = df['代码'].map(STOCK_SECTOR_MAP).fillna('')
            # 未匹配到的板块，兜底为热门板块（避免"其他"）
            df['板块'] = df['板块'].apply(lambda x: x if x in HOT_SECTORS else '人工智能')
        
        # 数值转换 + 资金单位转换（元→亿元）
        numeric_cols = ['最新价', '涨跌幅', '成交量', '成交额', '开盘价', '最高价', '最低价']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # 成交额转为亿元
        if '成交额' in df.columns:
            df['成交额'] = df['成交额'] / 1e8
        
        global_spot_cache = df
        cache_update_time = current_time
        return df
    except Exception as e:
        logger.error(f"获取行情数据失败：{str(e)}")
        # 返回兜底空DataFrame（确保列完整）
        return pd.DataFrame({
            '代码': [], '名称': [], '最新价': [], '涨跌幅': [], 
            '成交量': [], '成交额': [], '板块': []
        })

def get_real_time_market_summary():
    """市场情绪分析"""
    spot_df = get_spot_data_cached()
    if spot_df.empty:
        return pd.DataFrame({
            '上涨': [0], '下跌': [0], '平盘': [0],
            '赚钱效应': [0.5], '更新时间': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
    change_col = get_column_name_fixed(spot_df, ['涨跌幅', '涨跌幅%'])
    up_count = len(spot_df[spot_df[change_col] > 0.01])
    down_count = len(spot_df[spot_df[change_col] < -0.01])
    flat_count = len(spot_df) - up_count - down_count
    profit_effect = round(up_count / (up_count+down_count) if up_count+down_count>0 else 0.5, 2)
    return pd.DataFrame({
        '上涨': [up_count], '下跌': [down_count], '平盘': [flat_count],
        '赚钱效应': [profit_effect], '更新时间': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })

def get_real_time_board_flow():
    """板块资金流"""
    spot_df = get_spot_data_cached()
    if spot_df.empty:
        # 修复点：返回空数据时确保列完整
        return pd.DataFrame({'板块名称': [], '涨跌幅': [], '主力净流入': []})
    # 按板块分组计算
    sector_flow = spot_df.groupby('板块')['成交额'].sum().nlargest(8).reset_index()
    sector_flow.columns = ['板块名称', '主力净流入']
    sector_change = spot_df.groupby('板块')['涨跌幅'].mean().reset_index()
    sector_flow = sector_flow.merge(sector_change, left_on='板块名称', right_on='板块', how='left').fillna(0)
    return sector_flow[['板块名称', '涨跌幅', '主力净流入']]

def get_board_stocks(sector_name, spot_df, top_n=3):
    """获取板块龙头标的（修复数组长度不一致问题）"""
    try:
        if spot_df.empty:
            # 修复点2：确保默认数据长度严格匹配top_n
            default_codes = [k for k, v in STOCK_SECTOR_MAP.items() if v == sector_name][:top_n]
            # 不足top_n时补空值，确保长度一致
            while len(default_codes) < top_n:
                default_codes.append('000000')
            
            # 构造兜底DataFrame（所有列长度一致）
            return pd.DataFrame({
                '代码': default_codes,
                '名称': ['未知']*len(default_codes), 
                '最新价': [0]*len(default_codes), 
                '涨跌幅': [0]*len(default_codes), 
                '成交额': [0]*len(default_codes),
                '板块': [sector_name]*len(default_codes)
            })
        
        sector_stocks = spot_df[spot_df['板块'] == sector_name].copy()
        if sector_stocks.empty:
            # 同样确保默认数据长度匹配
            default_codes = [k for k, v in STOCK_SECTOR_MAP.items() if v == sector_name][:top_n]
            while len(default_codes) < top_n:
                default_codes.append('000000')
            sector_stocks = pd.DataFrame({
                '代码': default_codes,
                '名称': ['未知']*len(default_codes), 
                '最新价': [0]*len(default_codes), 
                '涨跌幅': [0]*len(default_codes), 
                '成交额': [0]*len(default_codes),
                '板块': [sector_name]*len(default_codes)
            })
        
        # 按涨跌幅排序取前N
        return sector_stocks.sort_values('涨跌幅', ascending=False).head(top_n)
    except Exception as e:
        logger.error(f"获取{sector_name}龙头股失败：{str(e)}")
        # 兜底返回长度为top_n的空数据
        return pd.DataFrame({
            '代码': ['000000']*top_n, '名称': ['未知']*top_n, 
            '最新价': [0]*top_n, '涨跌幅': [0]*top_n, 
            '成交额': [0]*top_n, '板块': [sector_name]*top_n
        })

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_real_time_stock_kline(stock_code):
    """K线数据"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(
            symbol=stock_code, period="daily", start_date=start_date,
            end_date=end_date, adjust="qfq"
        )
        col_map = {
            '收盘价格': '收盘', '最高价格': '最高', '最低价格': '最低',
            '开盘价格': '开盘', '成交量(手)': '成交量', '日期': '日期'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # 补全缺失列
        for col in ['日期', '开盘', '最高', '最低', '收盘', '成交量']:
            if col not in df.columns:
                df[col] = 0
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df['涨跌幅'] = df['收盘'].pct_change() * 100
        df['涨跌幅'] = df['涨跌幅'].fillna(0)
        return df
    except Exception as e:
        logger.error(f"获取{stock_code}K线数据失败：{str(e)}")
        return pd.DataFrame({
            '日期': [], '开盘': [], '最高': [], '最低': [], '收盘': [], 
            '成交量': [], '涨跌幅': []
        })

# ====================== 专业金融分析函数（保留原有逻辑） ======================
def calculate_all_tech_indicators(df):
    """完整技术指标计算"""
    df = df.copy()
    if df.empty:
        tech_cols = ['MA5', 'MA10', 'MA20', 'BOLL_MID', 'BOLL_UPPER', 'BOLL_LOWER',
                     'RSI14', 'EMA12', 'EMA26', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
                     'RSV', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'VOLATILITY', 'VOL5', 'VOL10']
        for col in tech_cols:
            df[col] = 0
        return df
    
    # 均线系统
    df['MA5'] = df['收盘'].rolling(window=5, min_periods=1).mean().fillna(0)
    df['MA10'] = df['收盘'].rolling(window=10, min_periods=1).mean().fillna(0)
    df['MA20'] = df['收盘'].rolling(window=20, min_periods=1).mean().fillna(0)
    # 布林带
    df['BOLL_MID'] = df['收盘'].rolling(window=20, min_periods=1).mean().fillna(0)
    boll_std = df['收盘'].rolling(window=20, min_periods=1).std().fillna(0.0001)
    df['BOLL_UPPER'] = df['BOLL_MID'] + 2 * boll_std
    df['BOLL_LOWER'] = df['BOLL_MID'] - 2 * boll_std
    # RSI
    delta = df['收盘'].diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().fillna(0.0001)
    df['RSI14'] = 100 - (100 / (1 + gain/loss)).fillna(50)
    # MACD
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False, min_periods=1).mean().fillna(0)
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False, min_periods=1).mean().fillna(0)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean().fillna(0)
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    # KDJ
    low_min = df['最低'].rolling(window=9, min_periods=1).min().fillna(0)
    high_max = df['最高'].rolling(window=9, min_periods=1).max().fillna(0.0001)
    df['RSV'] = ((df['收盘'] - low_min) / (high_max - low_min) * 100).fillna(50)
    df['KDJ_K'] = df['RSV'].ewm(span=3, adjust=False, min_periods=1).mean().fillna(0)
    df['KDJ_D'] = df['KDJ_K'].ewm(span=3, adjust=False, min_periods=1).mean().fillna(0)
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
    # 波动率+量能
    df['VOLATILITY'] = (df['最高'] - df['最低']).rolling(window=20, min_periods=1).mean().fillna(0)
    df['VOL5'] = df['成交量'].rolling(window=5, min_periods=1).mean().fillna(0)
    df['VOL10'] = df['成交量'].rolling(window=10, min_periods=1).mean().fillna(0)
    return df

def calculate_fibonacci_strategy(df):
    """斐波那契策略（建仓价≤现价）"""
    if df.empty:
        return {
            "当前价格": 0, "回撤位": {}, "拓展位": {}, "当前位置": "未知",
            "建仓建议": {"保守":0, "中性":0, "激进":0},
            "止盈建议": {"第一目标":0, "第二目标":0, "终极目标":0},
            "止损建议": {"绝对止损":0, "动态止损":0}, "波动率": 0
        }
    
    # 核心数据
    high_60d = df['最高'].tail(60).max()
    low_60d = df['最低'].tail(60).min()
    current_price = df['收盘'].iloc[-1]
    volatility = df['VOLATILITY'].iloc[-1]
    price_range = high_60d - low_60d
    rsi14 = df['RSI14'].iloc[-1] if 'RSI14' in df.columns else 50
    kdj_j = df['KDJ_J'].iloc[-1] if 'KDJ_J' in df.columns else 50
    
    # 斐波那契回撤/拓展位
    retracement = {
        0.000: round(high_60d, 2), 0.236: round(high_60d - price_range * 0.236, 2),
        0.382: round(high_60d - price_range * 0.382, 2), 0.500: round(high_60d - price_range * 0.500, 2),
        0.618: round(high_60d - price_range * 0.618, 2), 1.000: round(low_60d, 2)
    }
    extension = {
        1.000: round(high_60d, 2), 1.272: round(low_60d + price_range * 1.272, 2),
        1.618: round(low_60d + price_range * 1.618, 2)
    }
    
    # 位置判断
    if current_price >= retracement[0.236]:
        position = "超买区（强势）"
    elif current_price >= retracement[0.500]:
        position = "平衡区（震荡）"
    else:
        position = "超卖区（弱势）"
    
    # 建仓价计算（确保≤当前价）
    if rsi14 > 70 or kdj_j > 80:  # 超买：建仓价更低
        conservative_buy = round(retracement[0.618], 2)
        neutral_buy = round(retracement[0.500], 2)
        aggressive_buy = round(retracement[0.382], 2)
    elif rsi14 < 30 or kdj_j < 20:  # 超卖：建仓价接近现价
        conservative_buy = round(max(retracement[0.382], current_price - 0.5*volatility), 2)
        neutral_buy = round(max(retracement[0.236], current_price - 0.2*volatility), 2)
        aggressive_buy = round(current_price, 2)
    else:  # 震荡市
        conservative_buy = round(retracement[0.500], 2)
        neutral_buy = round(retracement[0.382], 2)
        aggressive_buy = round(min(retracement[0.236], current_price), 2)
    
    # 最终校验：所有建仓价≤当前价
    conservative_buy = min(conservative_buy, current_price)
    neutral_buy = min(neutral_buy, current_price)
    aggressive_buy = min(aggressive_buy, current_price)
    
    # 止盈/止损计算
    if position == "超买区（强势）":
        first_target = round(extension[1.000], 2)
        second_target = round(extension[1.272], 2)
        ultimate_target = round(extension[1.618], 2)
    else:
        first_target = round(retracement[0.000], 2)
        second_target = round(extension[1.000], 2)
        ultimate_target = round(extension[1.272], 2)
    
    absolute_stop = round(min(conservative_buy - 1.0*volatility, retracement[1.000]), 2)
    dynamic_stop = round(current_price - 1.5*volatility, 2)
    
    return {
        "当前价格": round(current_price, 2), "回撤位": retracement, "拓展位": extension,
        "当前位置": position,
        "建仓建议": {"保守": conservative_buy, "中性": neutral_buy, "激进": aggressive_buy},
        "止盈建议": {"第一目标": first_target, "第二目标": second_target, "终极目标": ultimate_target},
        "止损建议": {"绝对止损": absolute_stop, "动态止损": dynamic_stop},
        "波动率": round(volatility, 2)
    }

def generate_investment_view(tech_df, fund_flow, market_profit_effect):
    """生成投资观点（三层防护）"""
    invest_view = {
        "观点": "观望", "逻辑": "数据不足，无法分析", "标签": "🟠", "总分": 0
    }
    
    if tech_df.empty:
        return invest_view
    
    try:
        latest = tech_df.iloc[-1]
        fib_data = calculate_fibonacci_strategy(tech_df)
        position = fib_data.get("当前位置", "未知")
        
        # 技术指标评分
        tech_score = 0
        if latest['收盘'] > latest['MA20']: tech_score += 20
        if latest['MACD'] > latest['MACD_SIGNAL']: tech_score += 15
        if ANALYSIS_THRESHOLD['rsi_oversell'] < latest['RSI14'] < ANALYSIS_THRESHOLD['rsi_overbuy']: tech_score += 15
        if ANALYSIS_THRESHOLD['kdj_oversell'] < latest['KDJ_J'] < ANALYSIS_THRESHOLD['kdj_overbuy']: tech_score += 10
        if latest['成交量'] > latest['VOL10']: tech_score += 10
        
        # 资金面评分
        fund_score = 20 if fund_flow > ANALYSIS_THRESHOLD['fund_flow_positive'] else 0
        
        # 市场情绪评分
        market_score = 20 if market_profit_effect > ANALYSIS_THRESHOLD['赚钱效应'] else 5
        
        # 斐波那契位置评分
        fib_score = 0
        if "超卖区" in position: fib_score += 20
        elif "平衡区" in position: fib_score += 10
        
        total_score = tech_score + fund_score + market_score + fib_score
        
        # 观点生成
        if total_score >= 80:
            invest_view = {"观点":"买入", "标签":"🟢", "逻辑":f"趋势向上（收盘价>MA20）+ MACD金叉 + {position} + 主力净流入{fund_flow:.2f}亿元 + 市场赚钱效应{market_profit_effect}", "总分":total_score}
        elif total_score >= 60:
            invest_view = {"观点":"持有", "标签":"🟡", "逻辑":f"趋势中性 + 震荡指标正常 + {position} + 资金小幅流入 + 市场情绪中性", "总分":total_score}
        elif total_score >= 40:
            invest_view = {"观点":"观望", "标签":"🟠", "逻辑":f"趋势不明 + 震荡指标临界 + {position} + 资金流入不足 + 市场情绪一般", "总分":total_score}
        elif total_score >= 20:
            invest_view = {"观点":"减仓", "标签":"🔴", "逻辑":f"趋势向下（收盘价<MA20）+ MACD死叉 + {position} + 主力净流出{fund_flow:.2f}亿元 + 市场赚钱效应低", "总分":total_score}
        else:
            invest_view = {"观点":"清仓", "标签":"🔴🔴", "逻辑":f"趋势走弱 + 超买/超卖严重 + {position} + 资金大幅流出 + 市场情绪低迷", "总分":total_score}
            
    except Exception as e:
        logger.error(f"生成投资观点失败：{str(e)}")
    
    # 终极兜底
    required_keys = ["观点", "逻辑", "标签", "总分"]
    for key in required_keys:
        if key not in invest_view:
            invest_view[key] = "🟠" if key == "标签" else ("观望" if key == "观点" else (0 if key == "总分" else "数据异常，无法分析"))
    
    return invest_view

def plot_pro_tech_chart(stock_code, stock_name, df, fib_data):
    """专业技术分析图表"""
    if df.empty:
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "暂无数据", ha='center', va='center', fontsize=12)
        return fig
    
    df_plot = df.tail(60).copy()
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(8, 6), gridspec_kw={'height_ratios': [3,1,1], 'hspace':0.15})
    
    # K线+均线+斐波那契
    ax1.plot(df_plot['日期'], df_plot['收盘'], '#1f77b4', linewidth=1.2, label='收盘价')
    ax1.plot(df_plot['日期'], df_plot['MA5'], '#ff7f0e', linewidth=1, label='MA5')
    ax1.plot(df_plot['日期'], df_plot['MA20'], '#2ca02c', linewidth=1, label='MA20')
    ax1.plot(df_plot['日期'], df_plot['BOLL_UPPER'], 'gray', linestyle='--', linewidth=0.8, label='BOLL上轨')
    ax1.plot(df_plot['日期'], df_plot['BOLL_LOWER'], 'gray', linestyle='--', linewidth=0.8, label='BOLL下轨')
    
    # 斐波那契线
    if fib_data and '回撤位' in fib_data:
        for level, val in fib_data['回撤位'].items():
            ax1.axhline(y=val, color='gray', linestyle=':', alpha=0.6)
            ax1.text(df_plot['日期'].iloc[-1], val, f'{level}', fontsize=6)
    
    current_price = fib_data.get('当前价格', 0)
    ax1.scatter(df_plot['日期'].iloc[-1], current_price, color='red', s=30, label=f'实时价: {current_price}')
    ax1.set_title(f'{stock_code} {stock_name} 专业技术分析', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(alpha=0.2)
    
    # MACD
    macd_hist = df_plot['MACD_HIST'].fillna(0)
    colors = ['green' if x>0 else 'red' for x in macd_hist]
    ax2.bar(df_plot['日期'], macd_hist, color=colors, alpha=0.7, width=0.8)
    ax2.plot(df_plot['日期'], df_plot['MACD'], 'blue', linewidth=0.8, label='MACD')
    ax2.plot(df_plot['日期'], df_plot['MACD_SIGNAL'], 'orange', linewidth=0.8, label='Signal')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)
    
    # RSI+KDJ
    ax3.plot(df_plot['日期'], df_plot['RSI14'], '#9b59b6', linewidth=1, label='RSI14')
    ax3.plot(df_plot['日期'], df_plot['KDJ_K'], '#e67e22', linewidth=1, label='KDJ K')
    ax3.plot(df_plot['日期'], df_plot['KDJ_J'], '#e74c3c', linewidth=1, label='KDJ J')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.6)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.6)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.2)
    
    plt.tight_layout()
    return fig

# ====================== 主程序 ======================
def main():
    # 专业样式配置
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .stMetric {padding: 0.5rem !important;}
        .stDataFrame {font-size: 0.8rem !important;}
        .stExpander {margin-bottom: 0.5rem !important;}
    }
    .pro-header {font-size: 1.6rem; font-weight: bold; color: #1e3a8a;}
    .sector-header {font-size: 1.3rem; font-weight: bold; color: #3b82f6; margin-top: 1rem;}
    .view-tag {font-size: 1rem; font-weight: bold;}
    .metric-value {font-size: 1.2rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)
    
    # 页面标题
    st.markdown('<div class="pro-header">📊 专业股票分析系统（机构版）</div>', unsafe_allow_html=True)
    st.success(f"💡 数据更新至：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 资金单位：亿元")
    st.divider()
    
    if st.button("🚀 启动专业分析", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. 预加载数据
            status_text.text("预加载实时行情数据...")
            spot_df = get_spot_data_cached()
            progress_bar.progress(10)
            
            # 2. 市场情绪分析
            st.markdown('<div class="sector-header">📈 市场整体情绪</div>', unsafe_allow_html=True)
            status_text.text("分析市场情绪...")
            market_summary = get_real_time_market_summary()
            up = int(market_summary['上涨'].iloc[0])
            down = int(market_summary['下跌'].iloc[0])
            profit_effect = market_summary['赚钱效应'].iloc[0]
            
            # 情绪展示
            col1, col2, col3 = st.columns([1,1,1], gap="small")
            with col1: st.metric("上涨家数", up)
            with col2: st.metric("下跌家数", down)
            with col3: st.metric("赚钱效应", profit_effect)
            
            # 情绪判断
            if profit_effect > 0.6:
                st.info("✅ 市场情绪偏暖，赚钱效应良好，可积极操作")
            elif profit_effect > 0.4:
                st.warning("⚠️ 市场情绪中性，震荡为主，谨慎操作")
            else:
                st.error("❌ 市场情绪低迷，赚钱效应差，控制仓位")
            progress_bar.progress(20)
            
            # 3. 热点板块分析
            st.markdown('<div class="sector-header">🔥 核心热点板块（前8）</div>', unsafe_allow_html=True)
            status_text.text("分析热点板块资金流...")
            board_flow = get_real_time_board_flow()
            top8_sectors = board_flow['板块名称'].head(8).tolist() if not board_flow.empty else HOT_SECTORS[:8]
            
            # 板块资金流展示
            st.dataframe(
                board_flow.head(8).style.format({
                    '涨跌幅': '{:.2f}%',
                    '主力净流入': '{:.2f}亿元'
                }),
                use_container_width=True,
                height=320
            )
            progress_bar.progress(30)
            
            # 4. 板块龙头标的分析
            for sector_idx, sector_name in enumerate(top8_sectors):
                st.markdown(f'<div class="sector-header">🎯 {sector_idx+1}. {sector_name} 板块龙头分析</div>', unsafe_allow_html=True)
                status_text.text(f"分析{sector_name}板块...")
                
                # 获取板块3只龙头
                sector_stocks = get_board_stocks(sector_name, spot_df, top_n=3)
                
                # 逐个分析标的
                for stock_idx, (_, stock) in enumerate(sector_stocks.iterrows()):
                    # 兜底处理
                    stock_code = str(stock.get('代码', '')).zfill(6) if pd.notna(stock.get('代码')) else '000000'
                    stock_name = stock.get('名称', '未知') if pd.notna(stock.get('名称')) else '未知'
                    stock_price = round(stock.get('最新价', 0), 2) if pd.notna(stock.get('最新价')) else 0.0
                    stock_change = round(stock.get('涨跌幅', 0), 2) if pd.notna(stock.get('涨跌幅')) else 0.0
                    stock_fund = round(stock.get('成交额', 0), 2) if pd.notna(stock.get('成交额')) else 0.0
                    
                    with st.expander(f"【{stock_idx+1}】{stock_code} {stock_name}", expanded=True):
                        # 获取标的深度数据
                        kline_df = get_real_time_stock_kline(stock_code)
                        tech_df = calculate_all_tech_indicators(kline_df)
                        fib_data = calculate_fibonacci_strategy(tech_df)
                        invest_view = generate_investment_view(tech_df, stock_fund, profit_effect)
                        
                        # 展示核心信息
                        col1, col2 = st.columns([1, 1.5], gap="small")
                        with col1:
                            st.write("### 📊 核心数据")
                            st.write(f"- 最新价格：{stock_price} 元")
                            st.write(f"- 涨跌幅：{stock_change} %")
                            st.write(f"- 主力净流入：{stock_fund} 亿元")
                            view_tag = invest_view.get('标签', '🟠')
                            view_opinion = invest_view.get('观点', '观望')
                            view_logic = invest_view.get('逻辑', '数据异常，无法分析')
                            st.write(f"- **投资观点：{view_tag} {view_opinion}**")
                            st.write(f"- 观点逻辑：{view_logic}")
                            
                            st.write("### 🎯 建仓/止盈止损（基于斐波那契+技术指标）")
                            st.write(f"- 保守建仓：{fib_data.get('建仓建议', {}).get('保守', 0)} 元")
                            st.write(f"- 中性建仓：{fib_data.get('建仓建议', {}).get('中性', 0)} 元")
                            st.write(f"- 激进建仓：{fib_data.get('建仓建议', {}).get('激进', 0)} 元")
                            st.write(f"- 第一止盈：{fib_data.get('止盈建议', {}).get('第一目标', 0)} 元")
                            st.write(f"- 绝对止损：{fib_data.get('止损建议', {}).get('绝对止损', 0)} 元")
                        
                        with col2:
                            chart = plot_pro_tech_chart(stock_code, stock_name, tech_df, fib_data)
                            st.pyplot(chart)
                
                progress_bar.progress(30 + (sector_idx+1)*8)
            
            # 5. 总结建议
            st.markdown('<div class="sector-header">📋 整体操作建议</div>', unsafe_allow_html=True)
            st.success("""
            ### 专业操作建议
            1. **板块选择**：优先关注前8热点板块（资金流入多，活跃度高）；
            2. **标的选择**：选择「买入」评级标的，回避「减仓/清仓」评级标的；
            3. **建仓策略**：超买区逢低（保守/中性）建仓，超卖区可激进建仓；
            4. **风控策略**：严格执行斐波那契止损位，单笔仓位不超过总资金的10%；
            5. **止盈策略**：达到第一止盈位减仓50%，第二止盈位清仓。
            """)
            
            progress_bar.progress(100)
            st.success("✅ 专业分析完成！所有观点基于技术指标+斐波那契+资金面+市场情绪四维模型生成。")
        
        except Exception as e:
            logger.error(f"分析异常：{str(e)}\n{traceback.format_exc()}")
            st.error(f"运行异常：{str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
            plt.close('all')

if __name__ == "__main__":
    main()