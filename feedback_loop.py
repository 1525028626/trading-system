import pandas as pd
import baostock as bs
import datetime
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 引入公共库
from stock_common import Config, StockTransformer, DataProvider

JOURNAL_FILE = "ai_trading_journal.csv"

class FeedbackSystem:
    """
    预测反馈与验证系统 (Feedback Loop)
    功能:
    1. 盘前: 将 scan.py 生成的预测结果导入到日志文件 (ai_trading_journal.csv)
    2. 盘后: 验证预测结果与真实行情的偏差，更新日志状态
    3. 生成的数据用于后续的 "错题本精调" (Hard Example Mining)
    """
    def __init__(self):
        # 不再需要加载模型，因为直接读取 scan.py 的结果
        pass

    def record_predictions(self, model_type=None):
        """
        【步骤 1】盘前执行: 记录预测结果
        读取 scan_result_*.csv，筛选优质机会，存入交易日志。
        
        Args:
            model_type: 'conservative' 或 'aggressive' (如果为 None 则交互式选择)
        """
        print(f"\n{'='*40}\n📝 读取扫描结果 (Read Scan)\n{'='*40}")
        
        if model_type is None:
            print("请选择要导入的扫描结果:")
            print("1. 🛡️ 稳健模型结果 (scan_result_conservative.csv)")
            print("2. 🔥 激进模型结果 (scan_result_aggressive.csv)")
            choice = input("\n请输入数字 (1 或 2): ").strip()
            if choice == '2':
                model_type = "aggressive"
            else:
                model_type = "conservative"
        
        if model_type == 'aggressive':
            scan_file = "scan_result_aggressive.csv"
        else:
            scan_file = "scan_result_conservative.csv"
            
        if not os.path.exists(scan_file):
            print(f"❌ 未找到 {scan_file}，请先运行 scan.py 生成对应模式的扫描结果！")
            return

        # 初始化日志文件
        if not os.path.exists(JOURNAL_FILE):
            pd.DataFrame(columns=['date','model_type','code','start_price','pred_price','pred_pct','real_price','real_pct','error','status']).to_csv(JOURNAL_FILE, index=False)

        # 3. 读取扫描结果
        try:
            df_scan = pd.read_csv(scan_file)
            print(f"[*] 成功读取 {len(df_scan)} 条扫描记录")
        except Exception as e:
            print(f"❌ 读取扫描文件失败: {e}")
            return

        # === 新增：日期验证 ===
        if 'date' in df_scan.columns:
            # 检查第一条记录的日期
            scan_date = str(df_scan.iloc[0]['date'])
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            
            print(f"[*] 扫描数据日期: {scan_date} (今日: {today_str})")
            # 只有在交互模式下才询问，自动化模式下默认继续 (因为可能是复盘历史)
            if scan_date != today_str:
                print(f"⚠️ 警告: 扫描数据并非今日生成！(差异: {scan_date} vs {today_str})")
                # 只有当 model_type 是通过交互传入 None 时，才进行 input 确认
                # 但这里 model_type 可能被自动传入。
                # 简单起见，如果非交互模式 (调用时传入了 model_type)，则自动跳过询问
                # 但为了安全，我们还是仅打印警告，不阻塞。
                pass 
        else:
            print("⚠️ 警告: 扫描结果中未找到 'date' 列，将默认使用今日日期。")
            scan_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # 4. 筛选优质机会 (Top 20)
        if 'pct' not in df_scan.columns:
            print(f"❌ 扫描文件格式错误，缺少 'pct' 列。现有列: {list(df_scan.columns)}")
            return
            
        candidates = df_scan.sort_values(by='pct', ascending=False).head(20).copy()
        
        if candidates.empty:
            print("[-] 没有符合条件的优质机会")
            return

        # 5. 准备新记录
        records = []
        
        for _, row in candidates.iterrows():
            curr_price = row['price']
            pred_pct = row['pct']
            pred_price = curr_price * (1 + pred_pct / 100)
            
            # 优先使用扫描文件中的日期，如果没有则用统一的 scan_date
            row_date = row.get('date', scan_date)
            
            records.append({
                'date': row_date,
                'model_type': model_type,
                'code': row['code'],
                'start_price': curr_price,
                'pred_price': round(pred_price, 2),
                'pred_pct': round(pred_pct, 2),
                'real_price': 0.0,
                'real_pct': 0.0,
                'error': 0.0,
                'status': 'pending'
            })

        if not records:
            print("[-] 无有效数据。")
            return

        new_df = pd.DataFrame(records).sort_values(by='pred_pct', ascending=False).head(20)

        # 6. 安全写入日志 (读取-合并-保存，防止格式错乱)
        if os.path.exists(JOURNAL_FILE):
            try:
                existing_df = pd.read_csv(JOURNAL_FILE)
                
                # 兼容旧版本：如果旧文件没有 model_type 列，给它补上 'unknown'
                if 'model_type' not in existing_df.columns:
                    print("⚠️ 检测到旧版日志文件，正在自动升级结构...")
                    existing_df['model_type'] = 'unknown'
                
                # 检查是否存在当天的重复记录，如果存在则删除旧的，保留新的
                # 直接追加后去重，保留最新的记录（new_df 在后）
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df = combined_df.drop_duplicates(subset=['date', 'code', 'model_type'], keep='last')
            except:
                # 文件损坏或为空，直接覆盖
                final_df = new_df
        else:
            final_df = new_df

        final_df.to_csv(JOURNAL_FILE, index=False)
        print(f"✅ 已将 Top {len(new_df)} 预测结果同步到交易日志。")
        
        # 打印预览
        print("\n同步名单:")
        for _, item in new_df.iterrows():
            print(f"{item['code']} | 现价: {item['start_price']} -> 预测: {item['pred_price']:.2f} (涨幅 {item['pred_pct']}%) ")

    def verify_results(self):
        """
        【步骤 2】盘后执行: 验证预测结果
        自动获取最新收盘价，计算真实涨幅和预测误差。
        """
        if not os.path.exists(JOURNAL_FILE):
            print("❌ 无记录文件。")
            return
            
        print(f"\n{'='*40}\n🔍 收盘复盘验证 (Verify)\n{'='*40}")
        try:
            df = pd.read_csv(JOURNAL_FILE)
        except Exception as e:
            print(f"❌ 读取日志文件失败: {e}")
            return
        
        if 'status' not in df.columns:
            print("❌ 日志文件格式错误，缺少 'status' 列。")
            return

        pending_mask = df['status'] == 'pending'
        if not pending_mask.any():
            print("[-] 无待验证记录。")
            return
            
        updates = 0
        print(f"[*] 正在验证 {pending_mask.sum()} 条待处理记录...")
        
        for idx, row in df[pending_mask].iterrows():
            code = row['code']
            record_date = str(row['date']) # 确保是字符串
            
            # 获取最近 K 线 (必须足够长以通过 FeatureEngineer 的长度检查，至少30天)
            stock_df = DataProvider.fetch_stock_data(code, days=60)
            if stock_df is None or stock_df.empty:
                print(f"⚠️ 无法获取 {code} 的数据，跳过。")
                continue
            
            # 核心修复：直接在数据中查找 record_date 这一行
            # 注意：record_date 是预测的目标日期（通常是 T 日）
            # DataProvider 返回的 date 列应该是字符串格式 'YYYY-MM-DD'
            # target_row = stock_df[stock_df['date'] == record_date]
            
            # === 新增逻辑：查找 record_date 之后的第一个交易日 ===
            # 因为 record_date 是信号生成日(T)，我们需要验证的是 T+1 或之后的表现
            future_data = stock_df[stock_df['date'] > record_date]
            
            if not future_data.empty:
                # 找到了 T+1 日（或之后最近的一天）
                target_row = future_data.iloc[0]
                target_date = target_row['date']
                
                actual_close = target_row['close']
                start_price = row['start_price']
                
                # 计算真实涨幅
                real_pct = (actual_close - start_price) / start_price * 100
                error = abs(row['pred_pct'] - real_pct)
                
                # 更新记录
                df.at[idx, 'real_price'] = actual_close
                df.at[idx, 'real_pct'] = round(real_pct, 2)
                df.at[idx, 'error'] = round(error, 2)
                df.at[idx, 'status'] = 'verified'
                
                updates += 1
                status_icon = "✅" if error < 2.0 else "❌"
                print(f"{status_icon} {code} (信号:{record_date} -> 验证:{target_date}): 预测 {row['pred_pct']}% vs 真实 {real_pct:.2f}% | 误差 {error:.2f}")
            else:
                # 没找到 T+1 数据，可能是还没开盘/收盘
                # print(f"⏳ {code}: 尚未找到 {record_date} 之后的收盘数据")
                pass

        if updates > 0:
            df.to_csv(JOURNAL_FILE, index=False)
            print(f"\n✅ 成功验证并更新了 {updates} 条记录！")
        else:
            print("\n[-] 没有记录被更新。可能原因：\n1. 尚未收盘或数据未更新\n2. 记录日期非交易日 (如周末)\n3. DataProvider 网络问题")
            
            # 智能提示
            try:
                last_record_date = df[pending_mask]['date'].max()
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                print(f"[*] 调试信息: 待验证记录日期为 {last_record_date}，系统正在寻找该日期之后的行情数据。")
                print(f"[*] 当前系统日期: {today}")
            except: pass

if __name__ == "__main__":
    bs.login()
    sys = FeedbackSystem()
    action = input("1: 盘前记录 | 2: 盘后验证\n请输入: ")
    if action == '1': sys.record_predictions()
    elif action == '2': sys.verify_results()
    bs.logout()