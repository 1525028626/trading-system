import torch
import torch.nn as nn
import baostock as bs
import pandas as pd
import numpy as np
import math
import datetime
import os
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 数据处理器 (Data Processor)
# ==========================================
class DataProcessor:
    """
    数据预处理与标准化模块
    负责:
    1. 定义特征列和价格列
    2. 执行 Log Transformation
    3. 执行 StandardScaler 标准化
    4. 生成滑动窗口序列
    """
    # 必须与 Config.INPUT_DIM 保持一致 (自动计算)
    FEATURE_COLS = [
        'open','high','low','close','volume',
        'MA5','MA20','RSI','MACD',
        'K','D','J',
        'BOLL_MID','BOLL_UP','BOLL_LOW',
        'OBV','ATR','ICH_CONV','ICH_BASE',
        'market_close'
    ]
    
    # 需要进行 Log 变换的列 (价格和大幅值数据)
    PRICE_COLS = [
        'open','high','low','close','MA5','MA20',
        'BOLL_MID','BOLL_UP','BOLL_LOW',
        'ICH_CONV','ICH_BASE','market_close'
    ]

    @classmethod
    def get_input_dim(cls):
        return len(cls.FEATURE_COLS)

    @classmethod
    def preprocess_data(cls, df):
        """
        对 DataFrame 进行清洗和 Log 变换
        Returns:
            data_values: 预处理后的 numpy 数组
        """
        # 1. 检查列完整性
        missing_cols = [col for col in cls.FEATURE_COLS if col not in df.columns]
        if missing_cols:
            return None 
            
        data = df[cls.FEATURE_COLS].copy()
        
        # === 数值稳定性处理 ===
        # 将 inf 替换为 nan，然后 dropna
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        if len(data) < Config.LOOKBACK: return None
        
        # 2. Log Transform (价格类)
        # 增加 epsilon 和 maximum 保护，防止负数或0导致 nan/inf
        for col in cls.PRICE_COLS:
            # 确保输入 log 的值至少是 1e-5 (对于价格来说这几乎是0)
            # 即使原始数据是负数(虽然价格不应为负)，也会被截断为 1e-5，log结果为 -11.5
            data[col] = np.log(np.maximum(data[col], 1e-5))
            
        # 3. Log Transform (成交量与波动类)
        data['volume'] = np.log(np.maximum(data['volume'], 1e-5) + 1.0)
        
        # OBV 有负数，处理方式: log(abs(x)+1) * sign(x)
        # 这里 abs(x) 肯定是正的，+1 后肯定 > 1，log 安全
        data['OBV'] = np.log(np.abs(data['OBV']) + 1.0) * np.sign(data['OBV']) 
        
        # ATR 也是正数
        data['ATR'] = np.log(np.maximum(data['ATR'], 1e-5))
        
        # 再次检查是否有 nan (log 可能产生 nan 如果输入是 nan)
        if data.isnull().values.any():
            return None
            
        return data.values

    @classmethod
    def create_sequences(cls, data_values, lookback):
        """
        对数据进行标准化并生成序列 (用于训练)
        Args:
            data_values: 预处理后的 numpy 数组
            lookback: 窗口长度
        Returns:
            X, y 列表
        """
        scaler = StandardScaler()
        # 注意: 这里对整个传入的数据块进行 fit
        data_scaled = scaler.fit_transform(data_values)
        
        X, y = [], []
        # close 列在 FEATURE_COLS 中的索引
        close_idx = cls.FEATURE_COLS.index('close')
        
        for i in range(lookback, len(data_scaled)): # 预测下一个点，所以不需要 -1，如果是预测当前点则...
            # 等等，原来的逻辑是 range(LOOKBACK, len-1) ?
            # 原逻辑: all_y.append(data_scaled[i, 3]) 
            # 这里的 i 是当前要预测的时间点。
            # 序列是 [i-lookback : i] (不包含 i)
            # 所以输入是 t-N 到 t-1，输出是 t
            X.append(data_scaled[i-lookback : i])
            y.append(data_scaled[i, close_idx])
            
        return X, y

    @classmethod
    def prepare_inference_data(cls, df, lookback):
        """
        为预测准备数据 (用于 inference)
        Returns:
            input_tensor: 模型输入 [1, lookback, dim]
            scaler: 用于反归一化的 scaler 对象
        """
        data_values = cls.preprocess_data(df)
        if data_values is None: return None, None
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_values)
        
        input_seq = data_scaled[-lookback:]
        if len(input_seq) < lookback: return None, None
        
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        return input_tensor, scaler

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
class Config:
    """
    系统全局配置类
    """
    # 你的持仓配置 (格式: '代码': {'qty': 数量, 'cost': 成本价})
    MY_PORTFOLIO = {
        # 示例:
        # 'sh.600519': {'qty': 100, 'cost': 1800.00},
        # 'sz.000001': {'qty': 1000, 'cost': 12.50},
    }

    # --- 模型参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 自动获取维度，消除硬编码
    INPUT_DIM = DataProcessor.get_input_dim()  
    
    # Transformer 特有配置 (增强版)
    d_model = 128     # 嵌入维度 (64 -> 128)
    nhead = 8         # 多头注意力头数 (4 -> 8)
    num_layers = 4    # Encoder 层数 (3 -> 4)
    dropout = 0.2     # 增加 Dropout 防止过拟合
    
    OUTPUT_DIM = 1    # 输出维度 (预测价格)
    LOOKBACK = 60     # 时间窗口长度 (过去 60 天预测未来)
    EPOCHS = 40       # 训练轮数
    
    # 双核模型路径
    MODEL_PATH_CONSERVATIVE = "model_conservative.pth"
    MODEL_PATH_AGGRESSIVE = "model_aggressive.pth"
    # 默认模型路径 (为了兼容旧代码，指向稳健模型)
    MODEL_PATH = MODEL_PATH_CONSERVATIVE

# ==========================================
# 2. 核心模型: Transformer (优化版)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    位置编码层 (Positional Encoding)
    Transformer 没有循环结构，需要通过位置编码注入序列信息。
    使用正弦和余弦函数生成位置编码。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

class StockTransformer(nn.Module):
    """
    基于 Transformer Encoder 的股价预测模型
    结构:
    Input -> Linear -> PositionalEncoding -> TransformerEncoder -> Output MLP
    """
    def __init__(self):
        super(StockTransformer, self).__init__()
        
        # 1. 输入层: 将特征映射到 d_model 维度
        self.input_net = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, Config.d_model),
            nn.LayerNorm(Config.d_model)
        )
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(Config.d_model)
        
        # 3. Transformer 编码层 (开启 batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=Config.d_model, 
            nhead=Config.nhead, 
            dim_feedforward=256, 
            dropout=Config.dropout,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, Config.num_layers)
        
        # 4. 输出层: 预测下一个时间步的收盘价 (Scaled)
        self.output_net = nn.Sequential(
            nn.Linear(Config.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, Config.OUTPUT_DIM)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, feature]
        x = self.input_net(x)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出作为预测依据
        final_state = output[:, -1, :]
        
        prediction = self.output_net(final_state)
        return prediction

# ==========================================
# 3. 特征工程 (Feature Engineering)
# ==========================================
class FeatureEngineer:
    """
    特征工程类
    负责计算各种技术指标 (MA, RSI, MACD, KDJ, BOLL, OBV, ATR, Ichimoku)
    """
    @staticmethod
    def add_technical_indicators(df):
        """
        为 DataFrame 添加技术指标列
        Args:
            df: 包含 open, high, low, close, volume 的 DataFrame
        Returns:
            添加了指标的 DataFrame，并移除了 NaN 行
        """
        if len(df) < 30: return None
        # 1. 基础指标
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 2. RSI (相对强弱指标)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (异同移动平均线)
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        
        # 4. KDJ (随机指标)
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # 5. 布林带 (Bollinger Bands)
        df['BOLL_MID'] = df['close'].rolling(window=20).mean()
        df['BOLL_STD'] = df['close'].rolling(window=20).std()
        df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
        df['BOLL_LOW'] = df['BOLL_MID'] - 2 * df['BOLL_STD']
        
        # 6. OBV (能量潮)
        # 注意: 如果 volume 是字符串类型，会报错，必须确保是数值
        vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        close = pd.to_numeric(df['close'], errors='coerce').fillna(0)
        df['OBV'] = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        
        # 7. ATR (平均真实波幅) - 衡量波动率
        high = pd.to_numeric(df['high'], errors='coerce').fillna(0)
        low = pd.to_numeric(df['low'], errors='coerce').fillna(0)
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # 8. Ichimoku (一目均衡表) - 支撑阻力
        # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
        nine_period_high = high.rolling(window=9).max()
        nine_period_low = low.rolling(window=9).min()
        df['ICH_CONV'] = (nine_period_high + nine_period_low) / 2
        
        # Base Line (Kijun-sen): (26-period high + 26-period low)/2
        twenty_six_period_high = high.rolling(window=26).max()
        twenty_six_period_low = low.rolling(window=26).min()
        df['ICH_BASE'] = (twenty_six_period_high + twenty_six_period_low) / 2
        
        df.dropna(inplace=True)
        return df

# ==========================================
# 4. 数据提供者 (Data Provider)
# ==========================================
class DataProvider:
    """
    数据获取类
    负责与 Baostock API 交互，获取股票K线数据和大盘指数数据。
    支持本地文件缓存，加速二次启动。
    """
    _market_index_cache = None
    CACHE_DIR = "data_cache" # 缓存目录

    @classmethod
    def _ensure_cache_dir(cls):
        if not os.path.exists(cls.CACHE_DIR):
            os.makedirs(cls.CACHE_DIR)

    @classmethod
    def get_market_index(cls, days=500, end_date=None):
        """
        获取上证指数 (sh.000001) 数据作为市场环境特征
        """
        # 注意：如果指定了 end_date，缓存可能无效，这里简单起见，如果有 end_date 就不使用缓存
        if end_date is None and cls._market_index_cache is not None: 
            return cls._market_index_cache
            
        if end_date is None:
            end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            end_dt = datetime.datetime.now()
        else:
            end_date_str = end_date
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
        start_date = (end_dt - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        rs = bs.query_history_k_data_plus("sh.000001", "date,close", start_date=start_date, end_date=end_date_str, frequency="d")
        data = []
        while (rs.error_code == '0') & rs.next(): data.append(rs.get_row_data())
        
        df = pd.DataFrame(data, columns=['date', 'market_close'])
        df['market_close'] = pd.to_numeric(df['market_close'], errors='coerce')
        
        if end_date is None:
            cls._market_index_cache = df
            
        return df

    @classmethod
    def get_stock_list(cls, mode='conservative'):
        """
        获取股票列表
        """
        if mode == 'aggressive':
            print("[*] 正在获取全市场股票列表 (Aggressive Mode)...")
            # 使用 query_stock_basic 获取上市的股票列表
            rs = bs.query_stock_basic()
            codes = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                # row[0] is code, row[1] is name, row[4] is type (1=股票)
                # status: 1=上市
                if row[5] == '1': 
                    # 增强过滤: 剔除 ST 股
                    if 'ST' in row[1]: continue
                    codes.append(row[0])
            print(f"[*] 全市场股票池构建完毕 (已过滤ST): {len(codes)} 只")
            return codes
            
        else: # conservative
            print("[*] 正在获取核心资产股票列表 (HS300 + ZZ500)...")
            
            # 1. 沪深300
            rs_hs300 = bs.query_hs300_stocks()
            codes_hs300 = []
            while (rs_hs300.error_code == '0') & rs_hs300.next(): 
                codes_hs300.append(rs_hs300.get_row_data()[1])
                
            # 2. 中证500
            rs_zz500 = bs.query_zz500_stocks()
            codes_zz500 = []
            while (rs_zz500.error_code == '0') & rs_zz500.next(): 
                codes_zz500.append(rs_zz500.get_row_data()[1])
                
            # 3. 合并并去重
            codes = list(set(codes_hs300 + codes_zz500))
            print(f"[*] 成功构建核心股票池: {len(codes)} 只")
            return codes

    @staticmethod
    def get_all_stock_list_legacy():
        """(备用) 获取全市场所有股票"""
        print("[*] 正在获取全市场股票列表...")

    @classmethod
    def fetch_stock_data(cls, code, days=500, end_date=None):
        """
        获取单只股票的 K 线数据，并计算指标
        ** 优化版: 支持增量更新 **
        """
        import os
        cls._ensure_cache_dir()
        cache_path = os.path.join(cls.CACHE_DIR, f"{code}.csv")
        
        if end_date is None:
            end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            end_dt = datetime.datetime.now()
        else:
            end_date_str = end_date
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
        start_date = (end_dt - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        df = pd.DataFrame()
        need_download = True
        
        # 1. 尝试读取缓存
        if os.path.exists(cache_path):
            try:
                df_cache = pd.read_csv(cache_path)
                # 检查列是否完整
                if not df_cache.empty and 'date' in df_cache.columns and 'peTTM' in df_cache.columns:
                    last_cache_date = str(df_cache.iloc[-1]['date'])
                    first_cache_date = str(df_cache.iloc[0]['date'])
                    
                    # 确保数值列正确
                    numeric_cols = ['open','high','low','close','volume','turn','peTTM']
                    for col in numeric_cols:
                        if col in df_cache.columns:
                            df_cache[col] = pd.to_numeric(df_cache[col], errors='coerce')

                    # 情况A: 缓存完全满足要求 (足够新 且 足够长)
                    if last_cache_date >= end_date_str and first_cache_date <= start_date:
                        df = df_cache
                        need_download = False
                        
                    # 情况B: 缓存不够新，但足够长 -> 增量更新
                    elif first_cache_date <= start_date:
                        last_dt = datetime.datetime.strptime(last_cache_date, "%Y-%m-%d")
                        next_day = last_dt + datetime.timedelta(days=1)
                        
                        if next_day <= end_dt:
                            incr_start = next_day.strftime("%Y-%m-%d")
                            # print(f"[*] {code} 增量更新: {incr_start} ~ {end_date_str}")
                            rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,turn,peTTM", 
                                                            start_date=incr_start, end_date=end_date_str, 
                                                            frequency="d", adjustflag="2")
                            data_list = []
                            while (rs.error_code == '0') & rs.next(): data_list.append(rs.get_row_data())
                            
                            if data_list:
                                df_new = pd.DataFrame(data_list, columns=['date','open','high','low','close','volume','turn','peTTM'])
                                for col in numeric_cols:
                                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                                df = pd.concat([df_cache, df_new], ignore_index=True)
                                df.drop_duplicates(subset=['date'], keep='last', inplace=True)
                                df.to_csv(cache_path, index=False)
                                need_download = False
                            else:
                                # 没新数据 (可能休市)
                                df = df_cache
                                need_download = False
                        else:
                            df = df_cache
                            need_download = False
            except Exception as e:
                # print(f"Cache error: {e}")
                pass
            
        if need_download or df.empty:
            # 2. 网络下载 (全量)
            # adjustflag="2" 前复权
            rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,turn,peTTM", start_date=start_date, end_date=end_date_str, frequency="d", adjustflag="2")
            data_list = []
            while (rs.error_code == '0') & rs.next(): data_list.append(rs.get_row_data())
            if not data_list: return None
            
            df = pd.DataFrame(data_list, columns=['date','open','high','low','close','volume','turn','peTTM'])
            
            # 显式转换数值列
            numeric_cols = ['open','high','low','close','volume','turn','peTTM']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 更新缓存 (覆盖写入)
            df.to_csv(cache_path, index=False)
        
        # 3. 截取所需时间段
        mask = (df['date'] >= start_date) & (df['date'] <= end_date_str)
        df_final = df[mask].copy()
        
        if len(df_final) < 30: return None # 数据太短

        # 4. 计算指标 (Feature Engineering)
        df_final = FeatureEngineer.add_technical_indicators(df_final)
        if df_final is None: return None
        
        # 合并大盘指数
        market_df = cls.get_market_index(days=days, end_date=end_date)
        if market_df is not None and not market_df.empty:
            df_final = pd.merge(df_final, market_df, on='date', how='left')
            # 强化填充逻辑：前向填充 -> 后向填充 (处理开头缺失) -> 均值填充 (处理全空)
            df_final['market_close'] = df_final['market_close'].ffill().bfill()
            
            # 如果依然有 NaN (极少见，除非大盘数据完全没匹配上)，用 3000 点填充防止 dropna 删光数据
            if df_final['market_close'].isnull().any():
                df_final['market_close'].fillna(3000.0, inplace=True)
        else:
            # 如果获取不到大盘数据，给一个默认值，保证程序不崩
            df_final['market_close'] = 3000.0
            
        df_final.dropna(inplace=True)
        return df_final