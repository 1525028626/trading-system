import torch
import baostock as bs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from stock_common import Config, StockTransformer, DataProvider, DataProcessor

class StockScanner:
    """
    股票扫描与分析器
    功能:
    1. 加载训练好的 Transformer 模型
    2. 对单只或多只股票进行预测
    3. 分析持仓组合
    4. 扫描全市场挖掘潜力股
    """
    def __init__(self, model_type='conservative'):
        """
        初始化扫描器
        Args:
            model_type: 模型类型 ('conservative' 或 'aggressive')
        """
        self.model_type = model_type
        if model_type == 'aggressive':
            self.model_path = Config.MODEL_PATH_AGGRESSIVE
        else:
            self.model_path = Config.MODEL_PATH_CONSERVATIVE
            
        self.model = StockTransformer().to(Config.DEVICE)
        if os.path.exists(self.model_path):
            # 加载时加上 map_location 防止 GPU/CPU 冲突
            self.model.load_state_dict(torch.load(self.model_path, map_location=Config.DEVICE))
            self.model.eval()
            print(f"[*] 模型加载成功: {self.model_path} ({model_type})")
        else:
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}！请先运行 train.py 训练对应模式。")

    def predict(self, df):
        """
        对单只股票进行预测
        Args:
            df: 包含历史数据的 DataFrame (长度需 >= LOOKBACK)
        Returns:
            pred_price: 预测的下一个交易日收盘价 (绝对值)
        """
        feature_cols = DataProcessor.FEATURE_COLS
        
        # 检查列是否存在
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少列: {missing_cols}")
            
        # === 使用 DataProcessor 统一处理 ===
        input_tensor, scaler = DataProcessor.prepare_inference_data(df, Config.LOOKBACK)
        if input_tensor is None:
            raise ValueError("数据不足以构建序列")
            
        input_tensor = input_tensor.to(Config.DEVICE)
        
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).item()
            
        # 反归一化: 只需要还原 close
        # 注意：这里还原出来的是 Log Price (Scaled 后的) -> Scaler Inverse -> Log Price -> Exp -> Price
        dummy = np.zeros((1, Config.INPUT_DIM))
        dummy[0, feature_cols.index('close')] = pred_scaled 
        
        # 1. Inverse Scale
        pred_log_price = scaler.inverse_transform(dummy)[0, feature_cols.index('close')]
        
        # 2. Exp (还原 Log)
        pred_price = np.exp(pred_log_price)
        
        return pred_price

    def analyze_portfolio(self):
        """
        持仓智能分析功能
        读取 Config.MY_PORTFOLIO 中的持仓，逐个进行预测和诊断
        """
        print(f"\n{'='*40}\n💼 持仓智能分析\n{'='*40}")
        print(f"{'代码':<10} {'现价':<8} {'AI预测':<8} {'预期%':<8} {'RSI':<6} {'建议':<10} {'仓位'}")
        print("-" * 80)
        
        for code, info in Config.MY_PORTFOLIO.items():
            cost = info['cost']
            try:
                df = DataProvider.fetch_stock_data(code, days=200)
                if df is None: continue
                
                curr = df.iloc[-1]['close']
                rsi = df.iloc[-1]['RSI']
                # turn = df.iloc[-1]['turn'] # 新增换手率查看
                
                pred = self.predict(df)
                pred_pct = (pred - curr) / curr * 100
                profit_pct = (curr - cost) / cost * 100
                
                # === 动态仓位管理 (Position Sizing) ===
                # 逻辑：结合 预测涨幅(pct) 和 技术指标置信度(RSI)
                # 1. 基础建议
                action = "🟢 持有"
                pos_suggest = "0%"

                # 2. 止损/卖出逻辑
                if profit_pct <= -5.0:
                    action = "🛑 止损"
                    pos_suggest = "清仓"
                elif pred_pct < -1.0 and rsi > 70:
                    action = "🔴 卖出"
                    pos_suggest = "减仓/清仓"
                
                # 3. 买入/加仓逻辑 (基于置信度)
                elif pred_pct > 0:
                    # 场景A: 极高确定性 (预测涨幅>5% 且 RSI低位<30) -> 重仓
                    if pred_pct > 5.0 and rsi < 30:
                        action = "🚀 强力买入"
                        pos_suggest = "50%+"
                    # 场景B: 高确定性 (预测涨幅>3% 且 RSI健康<70) -> 中仓
                    elif pred_pct > 3.0 and rsi < 70:
                        action = "🔵 买入"
                        pos_suggest = "30%"
                    # 场景C: 一般确定性 (预测涨幅>1%) -> 轻仓
                    elif pred_pct > 1.0:
                        action = "⚪ 增持"
                        pos_suggest = "10%"
                    else:
                        action = "🟡 观望"
                        pos_suggest = "0%"
                
                # 格式化输出
                print(f"{code:<10} {curr:<8.2f} {pred:<8.2f} {pred_pct:<8.2f} {rsi:<6.1f} {action:<10} {pos_suggest}")
            except: pass

    def scan_all(self):
        """
        全市场扫描功能
        扫描 Config 中定义的股票池，筛选出高潜力股
        结果保存为 CSV 文件
        """
        print(f"\n{'='*40}\n🔭 {self.model_type} 模式全市场扫描 (Input Dim: {Config.INPUT_DIM})\n{'='*40}")
        all_stocks = DataProvider.get_stock_list(mode=self.model_type)
        results = []
        
        print(f"[*] 正在扫描 {len(all_stocks)} 只股票...")
        error_count = 0
        
        for code in tqdm(all_stocks):
            try:
                df = DataProvider.fetch_stock_data(code, days=200)
                if df is None or len(df) < Config.LOOKBACK: continue
                
                pred = self.predict(df)
                curr = df.iloc[-1]['close']
                curr_date = df.iloc[-1]['date']
                
                # 获取辅助指标用于筛选
                rsi = df.iloc[-1]['RSI']
                
                # === 垃圾股过滤 ===
                # 1. 过滤亏损股 (PE < 0)
                if 'peTTM' in df.columns:
                    pe = df.iloc[-1]['peTTM']
                    if pe < 0: continue # 亏损
                    
                # 2. 过滤僵尸股 (换手率 < 0.5% 或成交量过低)
                if 'turn' in df.columns:
                    turn = df.iloc[-1]['turn']
                    if turn < 0.5: continue # 极低流动性
                
                pct = (pred - curr) / curr * 100
                
                # 过滤异常值
                if abs(pct) < 20: 
                    results.append({'code': code, 'date': curr_date, 'price': curr, 'pct': pct, 'rsi': rsi})
                else:
                    if error_count < 3:
                        # print(f"[Debug] {code} 预测涨幅异常: {pct:.2f}% (Pred: {pred:.2f}, Curr: {curr:.2f})")
                        error_count += 1
                        
            except Exception as e:
                if error_count < 3:
                    print(f"[Error] {code} 扫描失败: {e}")
                    error_count += 1
                continue
            
        results.sort(key=lambda x: x['pct'], reverse=True)
        
        print("\n🔥 潜力榜 Top 10:")
        print(f"{'代码':<10} {'现价':<8} {'预期涨幅':<10} {'RSI':<6}")
        print("-" * 50)
        
        for item in results[:10]:
            icon = "🚀" if item['pct'] > 3.0 else "📈"
            print(f"{item['code']:<10} {item['price']:<8.2f} {icon} {item['pct']:<6.2f}% {item['rsi']:<6.1f}")
            
        filename = f"scan_result_{self.model_type}.csv"
        # 显式指定列名，防止结果为空时无法生成Header
        df_res = pd.DataFrame(results, columns=['code', 'date', 'price', 'pct', 'rsi'])
        df_res.to_csv(filename, index=False)
        print(f"\n[*] 结果已保存至 {filename} (共 {len(df_res)} 条)")

if __name__ == "__main__":
    bs.login()
    print(f"\n{'='*50}\n🤖 AI 选股助手 (Scanner Console)\n{'='*50}")
    print("请选择功能:")
    print("1. 💼 持仓智能分析 (Portfolio Analysis)")
    print("2. 🔭 市场扫描 (Market Scan)")
    func_choice = input("\n请输入数字 (1 或 2): ").strip()

    print("\n请选择模型核心:")
    print("1. 🛡️ 稳健模型 (Conservative): 适合防守/白马/ETF")
    print("2. 🔥 激进模型 (Aggressive): 适合博弈/题材/妖股")
    model_choice = input("\n请输入数字 (1 或 2): ").strip()
    
    model_type = 'aggressive' if model_choice == '2' else 'conservative'
    
    try:
        scanner = StockScanner(model_type=model_type)
        
        if func_choice == '1':
            scanner.analyze_portfolio()
        elif func_choice == '2':
            scanner.scan_all()
        else:
            print("无效选择")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        
    bs.logout()