import torch
import torch.nn as nn
import torch.optim as optim
import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
import datetime
from dateutil.relativedelta import relativedelta

# 引入公共库
from stock_common import Config, StockTransformer, DataProvider, DataProcessor

def train_epoch_for_backtest(model, codes, end_date, learning_rate=0.0002, epochs=5):
    """
    针对特定截止日期的微调训练函数 (Incremental Learning)
    """
    print(f"[*] [Backtest] 正在针对日期 {end_date} 进行微调训练...")
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    all_X, all_y = [], []
    
    # 为了速度，回溯训练时只取部分股票或减少回看天数
    # 这里我们随机取 50 只股票进行代表性训练，或者全量训练（取决于计算资源）
    # 假设资源有限，每次随机抽 50 只核心资产
    sample_codes = random.sample(codes, min(len(codes), 50))
    
    valid_count = 0
    for code in sample_codes:
        # 获取截至 end_date 的数据
        # fetch_days 不需要太长，只要覆盖 LOOKBACK + 一些训练样本即可
        # 比如 200 天，大约有 200 - 60 = 140 个样本
        df = DataProvider.fetch_stock_data(code, days=200, end_date=end_date)
        if df is None or len(df) < Config.LOOKBACK + 10: continue
        
        # === 使用 DataProcessor 统一处理 ===
        data_values = DataProcessor.preprocess_data(df)
        if data_values is None: continue
        
        # 构造滑动窗口样本
        X_batch, y_batch = DataProcessor.create_sequences(data_values, Config.LOOKBACK)
        all_X.extend(X_batch)
        all_y.extend(y_batch)
            
        valid_count += 1

    if not all_X:
        print("[-] 没有有效数据，跳过本次训练")
        return model

    # 划分训练集和验证集 (80% / 20%)
    split_idx = int(len(all_X) * 0.8)
    X_train_np = np.array(all_X[:split_idx])
    y_train_np = np.array(all_y[:split_idx])
    X_val_np = np.array(all_X[split_idx:])
    y_val_np = np.array(all_y[split_idx:])

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(Config.DEVICE)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(Config.DEVICE)
    
    if len(X_val_np) > 0:
        X_val = torch.tensor(X_val_np, dtype=torch.float32).to(Config.DEVICE)
        y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1).to(Config.DEVICE)
    else:
        X_val = None
    
    batch_size = 128
    model.train()
    
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0
        total_acc = 0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            optimizer.zero_grad()
            out = model(X_train[indices])
            
            # === Loss 改进 ===
            base_loss = criterion(out, y_train[indices])
            close_idx = DataProcessor.FEATURE_COLS.index('close')
            last_close = X_train[indices, -1, close_idx].view(-1, 1)
            diff_pred = out - last_close
            diff_real = y_train[indices] - last_close
            penalty = torch.where(diff_pred * diff_real < 0, base_loss * 2.0, torch.zeros_like(base_loss))
            loss = base_loss + penalty.mean()
            
            # Acc
            acc = ((diff_pred * diff_real) > 0).float().mean().item()
            total_acc += acc
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
    # === 验证集评估 ===
    val_acc_str = "N/A"
    if X_val is not None:
        model.eval()
        with torch.no_grad():
            out_val = model(X_val)
            # Val Acc
            last_close_val = X_val[:, -1, close_idx].view(-1, 1)
            diff_pred_val = out_val - last_close_val
            diff_real_val = y_val - last_close_val
            val_acc = ((diff_pred_val * diff_real_val) > 0).float().mean().item()
            val_acc_str = f"{val_acc*100:.2f}%"
            
    print(f"    -> Train Loss: {total_loss/len(X_train)*batch_size:.6f} | Train Acc: {total_acc/len(X_train)*batch_size*100:.2f}% | Val Acc: {val_acc_str}")
    return model

def run_backtest_training():
    """
    历史回溯训练主程序
    功能:
    1. 设定回溯时间轴 (Start Date -> End Date)
    2. 逐步推进时间，模拟"每月更新模型"
    3. 支持"从头训练"或"微调现有模型"
    """
    print(f"\n{'='*50}\n⏳ 历史回溯训练系统 (Backtrack Training)\n{'='*50}")
    
    # 1. 设置回溯范围
    start_date_str = input("请输入回溯开始日期 (例如 2023-01-01): ").strip()
    months = int(input("请输入回溯持续月数 (例如 12): ").strip())
    
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    except:
        print("❌ 日期格式错误，请使用 YYYY-MM-DD")
        return

    print("\n请选择模型核心:")
    print("1. 🛡️ 稳健模型 (Conservative)")
    print("2. 🔥 激进模型 (Aggressive)")
    model_choice = input("请输入数字 (1 或 2): ").strip()
    
    if model_choice == '2':
        model_type = 'aggressive'
        model_path = Config.MODEL_PATH_AGGRESSIVE
    else:
        model_type = 'conservative'
        model_path = Config.MODEL_PATH_CONSERVATIVE
        
    bs.login()
    
    # 2. 初始化模型
    model = StockTransformer().to(Config.DEVICE)
    
    print("\n请选择训练模式:")
    print("1. 🐣 从头训练 (From Scratch): 忽略现有模型，从回溯开始日期重新训练")
    print("2. 🧠 微调现有模型 (Fine-tune Existing): 加载现有模型，在此基础上进行回溯训练")
    mode_choice = input("请输入数字 (1 或 2): ").strip()
    
    if mode_choice == '2' and os.path.exists(model_path):
        print(f"[*] 加载现有模型 {model_path} 作为起点...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("将自动切换为从头训练模式...")
    else:
        if mode_choice == '2':
            print(f"⚠️ 未找到现有模型 {model_path}，将初始化新模型...")
        else:
            print("[*] 初始化新模型作为起点...")

    # 获取股票列表 (只获取一次)
    codes = DataProvider.get_stock_list(mode=model_type)
    
    # 3. 时间循环
    current_date = start_date
    for i in range(months):
        current_date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n>>> [Step {i+1}/{months}] 模拟日期: {current_date_str}")
        
        # 执行微调训练
        # 模拟在这个日期，我们只能看到过去的数据，并基于此更新模型
        model = train_epoch_for_backtest(model, codes, current_date_str)
        
        # 保存中间检查点 (可选)
        # checkpoint_path = f"checkpoint_{current_date_str}.pth"
        # torch.save(model.state_dict(), checkpoint_path)
        
        # 推进时间 (每月)
        current_date = current_date + relativedelta(months=1)
        
    # 4. 保存最终模型
    print(f"\n{'='*50}")
    save_choice = input(f"回溯训练完成。是否覆盖原模型 {model_path}? (y/n): ")
    if save_choice.lower() == 'y':
        torch.save(model.state_dict(), model_path)
        print(f"✅ 模型已更新并保存至 {model_path}")
    else:
        backup_path = f"{model_path}.backtest_final.pth"
        torch.save(model.state_dict(), backup_path)
        print(f"✅ 模型已另存为 {backup_path}")
        
    bs.logout()

if __name__ == "__main__":
    run_backtest_training()
