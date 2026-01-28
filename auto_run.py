import time
import datetime
import baostock as bs
from train import train_model
from scan import StockScanner
from feedback_loop import FeedbackSystem

def auto_run():
    print(f"\n{'='*60}")
    print(f"🤖 AI 自动交易闭环系统 (Auto-Loop System)")
    print(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 登录 BS
    bs.login()
    
    try:
        # ==========================================
        # 1. 验证阶段 (Verification)
        # ==========================================
        print("\n>>> [Step 1/5] 验证昨日预测 (Feedback Verification)")
        feedback_sys = FeedbackSystem()
        feedback_sys.verify_results()
        
        # ==========================================
        # 2. 微调阶段 (Fine-tuning)
        # ==========================================
        print("\n>>> [Step 2/5] 错题本增量微调 (Incremental Fine-tuning)")
        # 只有在有错题被验证后，微调才有意义，但程序会自动检查错题本
        # 对两个模型分别进行微调
        
        print("\n--- 正在微调 [稳健模型] ---")
        train_model(model_type='conservative', mode='2') # mode='2' 是增量模式
        
        print("\n--- 正在微调 [激进模型] ---")
        train_model(model_type='aggressive', mode='2')

        # ==========================================
        # 3. 持仓分析 (Portfolio Analysis)
        # ==========================================
        print("\n>>> [Step 3/5] 持仓智能分析 (Portfolio Analysis)")
        
        # 实例化扫描器
        # 注意：这里需要重新实例化，因为之前的扫描器可能没有加载最新的模型（虽然在这里模型刚微调完，重载是好的）
        scanner_con = StockScanner(model_type='conservative')
        scanner_agg = StockScanner(model_type='aggressive')
        
        print("\n--- 🛡️ 稳健模型持仓建议 ---")
        scanner_con.analyze_portfolio()
        
        print("\n--- 🔥 激进模型持仓建议 ---")
        scanner_agg.analyze_portfolio()
        
        # ==========================================
        # 4. 扫描阶段 (Scanning)
        # ==========================================
        print("\n>>> [Step 4/5] 全市场扫描 (Market Scanning)")
        
        print("\n--- 正在扫描 [稳健模型] (HS300+ZZ500) ---")
        scanner_con.scan_all()
        
        print("\n--- 正在扫描 [激进模型] (All Market) ---")
        scanner_agg.scan_all()
        
        # ==========================================
        # 5. 记录阶段 (Recording)
        # ==========================================
        print("\n>>> [Step 5/5] 记录预测结果 (Recording Predictions)")
        # 自动将扫描结果写入日志
        
        print("\n--- 记录 [稳健模型] 结果 ---")
        feedback_sys.record_predictions(model_type='conservative')
        
        print("\n--- 记录 [激进模型] 结果 ---")
        feedback_sys.record_predictions(model_type='aggressive')
        
        print(f"\n{'='*60}")
        print("✅ 自动化流程执行完毕！")
        print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ 自动化流程异常中断: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bs.logout()

if __name__ == "__main__":
    auto_run()
