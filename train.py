import torch
import torch.nn as nn
import torch.optim as optim
import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random

# 引入公共库
from stock_common import Config, StockTransformer, DataProvider, DataProcessor

from torch.utils.data import Dataset, DataLoader

# 定义 worker 初始化函数，确保子进程能登录 baostock
def worker_init_fn(worker_id):
    bs.login() 

class StockDataset(Dataset):
    def __init__(self, codes, train_mode, fetch_days, lookback):
        self.codes = codes
        self.train_mode = train_mode
        self.fetch_days = fetch_days
        self.lookback = lookback
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = self.codes[idx]
        
        # 1. 获取数据 (增量更新逻辑在 fetch_stock_data 内部处理)
        # 注意: 这里调用的是 stock_common.py 里的 fetch_stock_data
        df = DataProvider.fetch_stock_data(code, days=self.fetch_days)
        if df is None or len(df) < self.lookback + 5:
            return [], []
            
        # 2. 增量模式下的特殊过滤 (僵尸股/停牌)
        if self.train_mode == '2':
            # 停牌检查
            try:
                last_dt = pd.to_datetime(str(df.iloc[-1]['date']))
                if (pd.Timestamp.now() - last_dt).days > 5:
                    return [], []
            except: pass
            
            # 成交量检查
            try:
                recent_vol = pd.to_numeric(df['volume'].tail(5)).mean()
                if recent_vol < 10000: # 日均成交不足
                    return [], []
            except: pass
        
        # 3. 预处理
        data_values = DataProcessor.preprocess_data(df)
        if data_values is None: return [], []
        
        # 4. 生成序列
        X_batch, y_batch = DataProcessor.create_sequences(data_values, self.lookback)
        
        # 简单 NaN 检查
        if np.isnan(X_batch).any() or np.isnan(y_batch).any():
            return [], []
            
        return X_batch, y_batch

# 将多个 batch (每个是 [X_list, y_list]) 合并为一个大列表
def collate_fn(batch):
    X_all = []
    y_all = []
    for X, y in batch:
        if X: # X 不为空列表
            X_all.extend(X)
            y_all.extend(y)
    return X_all, y_all

JOURNAL_FILE = "ai_trading_journal.csv"

def get_feedback_data(model_type='conservative'):
    """
    读取错题本，构建针对性训练数据 (Hard Example Mining)
    根据当前训练的模型类型，只提取对应类型的错题
    
    Args:
        model_type: 'conservative' 或 'aggressive'
        
    Returns:
        fb_X: 错题样本的特征序列列表
        fb_y: 错题样本的真实目标值列表
    """
    if not os.path.exists(JOURNAL_FILE):
        return [], []

    df = pd.read_csv(JOURNAL_FILE)
    
    # 兼容旧版 CSV (没有 model_type 列的情况)
    if 'model_type' not in df.columns:
        print("⚠️ 错题本未包含模型类型信息，将跳过筛选（可能导致模型混淆）")
        # 筛选出已经验证过且误差较大的记录 (Error > 3.0%)
        mistakes = df[(df['status'] == 'verified') & (df['error'] > 3.0)]
    else:
        # 筛选: verified + error > 3.0% + model_type 匹配
        mistakes = df[
            (df['status'] == 'verified') & 
            (df['error'] > 3.0) & 
            (df['model_type'] == model_type)
        ]
    
    if mistakes.empty: return [], []

    print(f"[*] 错题本加载 ({model_type}): 发现 {len(mistakes)} 个严重错误，准备提取数据...")
    
    fb_X, fb_y = [], []
    for _, row in tqdm(mistakes.iterrows(), total=len(mistakes)):
        code = row['code']
        error_date = row['date']
        
        # 获取足够长的数据以构建序列
        df = DataProvider.fetch_stock_data(code, days=Config.LOOKBACK + 100)
        if df is None: continue
        
        # 找到错题发生的日期
        target_rows = df[df['date'] == error_date]
        if target_rows.empty:
            print(f"[-] {code}: 未找到日期 {error_date} (数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
            continue
        
        # 核心修复: 获取该行在 df 中的位置 (0-based integer position)，而不是 Label Index
        target_label_idx = target_rows.index[0]
        
        # === 使用 DataProcessor 统一处理 ===
        # 注意：这里可能会因为清洗 Inf 而删除行，导致长度变化
        data_values = DataProcessor.preprocess_data(df)
        if data_values is None: 
            print(f"[-] {code}: 预处理失败 (可能数据全空)")
            continue
        
        # 重新定位 target_pos (在 data_values 中的位置)
        # 我们需要模拟 preprocess_data 的清洗过程来找到对应关系
        # 或者更简单：DataProcessor.preprocess_data 其实是基于 FEATURE_COLS 清洗的
        
        # 1. 复现清洗逻辑找到保留下来的 Index
        temp_df = df[DataProcessor.FEATURE_COLS].copy()
        temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_indices = temp_df.dropna().index
        
        # 2. 检查我们的目标行是否还在
        if target_label_idx not in valid_indices:
            print(f"[-] {code}: 目标行 {error_date} 因包含 Inf/NaN 被清洗，跳过")
            continue
            
        # 3. 获取新的位置 (0-based index in data_values)
        try:
            target_pos = valid_indices.get_loc(target_label_idx)
        except:
            # 兼容性写法
            target_pos = list(valid_indices).index(target_label_idx)
            
        if target_pos < Config.LOOKBACK: 
            print(f"[-] {code}: 历史数据不足 ({target_pos} < {Config.LOOKBACK})")
            continue

        # 注意: preprocess_data 已经做了 Log 变换，现在需要 Scale 并构建序列
        # 我们只关心 target_pos 这个点的预测
        # create_sequences 会返回所有可能的序列，我们只取 target_pos 对应的那个
        
        # 暂时手动处理以精确定位 target_pos
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_values)
        
        # 序列: [target_pos - LOOKBACK : target_pos]
        seq_x = data_scaled[target_pos - Config.LOOKBACK : target_pos]
        seq_y = data_scaled[target_pos, DataProcessor.FEATURE_COLS.index('close')]
        
        fb_X.append(seq_x)
        fb_y.append(seq_y)
        
    return fb_X, fb_y

def train_model(model_type=None, mode=None):
    """
    主训练流程
    包含:
    1. 模式选择 (稳健/激进)
    2. 训练类型选择 (全量/增量)
    3. Phase 1: 大规模基础训练
    4. Phase 2: 错题本精调
    
    Args:
        model_type: 'conservative' 或 'aggressive' (如果为 None 则交互式选择)
        mode: '1' (全量) 或 '2' (增量) (如果为 None 则交互式选择)
    """
    print(f"\n{'='*50}\n🚀 AI 训练控制台 (Training Console)\n{'='*50}")
    
    # 1. 选择模型类型
    if model_type is None:
        print("请选择模型类型:")
        print("1. 🛡️ 稳健模型 (Conservative): 仅训练 HS300+ZZ500 (核心资产)")
        print("2. 🔥 激进模型 (Aggressive): 训练全市场股票 (包含题材/小票)")
        model_type_input = input("\n请输入数字 (1 或 2): ").strip()
        model_type = 'aggressive' if model_type_input == '2' else 'conservative'
    
    if model_type == 'aggressive':
        model_path = Config.MODEL_PATH_AGGRESSIVE
        print(f"\n[!] 已选择: 🔥 激进模型 (保存路径: {model_path})")
    else:
        model_path = Config.MODEL_PATH_CONSERVATIVE
        print(f"\n[!] 已选择: 🛡️ 稳健模型 (保存路径: {model_path})")

    # 2. 选择训练模式
    if mode is None:
        print("\n请选择训练模式:")
        print("1. 🆕 全量训练 (Full Train): 删除旧模型，从零开始学习 (适合周末/大规模更新)")
        print("2. 🧠 增量精调 (Incremental): 加载旧模型，只学新数据和错题 (适合每日收盘后)")
        mode = input("\n请输入数字 (1 或 2): ").strip()
    
    # === 配置训练参数 ===
    if mode == '1':
        print("\n[!] 已选择: 全量训练模式")
        if os.path.exists(model_path):
            print(f"[*] 删除旧模型 {model_path} ...")
            try: os.remove(model_path)
            except: pass
        
        # 全量训练参数：高学习率，多轮次
        LEARNING_RATE = 0.001 
        EPOCHS = Config.EPOCHS # 默认 40
        load_existing = False
        
    elif mode == '2':
        print("\n[!] 已选择: 增量精调模式")
        if not os.path.exists(model_path):
            print(f"❌ 错误：未找到现有模型 {model_path}，无法进行增量训练！请先选择全量训练。")
            return
            
        # 增量训练参数：低学习率，少轮次 (防止遗忘)
        LEARNING_RATE = 0.0002 
        EPOCHS = 10 
        load_existing = True
        
    else:
        print("无效输入，退出。")
        return

    bs.login()
    
    # 初始化模型
    # 注意：因为 INPUT_DIM 可能变化，如果加载旧模型形状不匹配会报错
    model = StockTransformer().to(Config.DEVICE)
    
    if load_existing:
        print(f"[*] 正在加载现有模型权重: {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=Config.DEVICE)
            
            # 检查 input_net.0.weight 的形状是否匹配当前 Config.INPUT_DIM
            if state_dict['input_net.0.weight'].shape[1] != Config.INPUT_DIM:
                print(f"⚠️ 模型输入维度不匹配 (旧: {state_dict['input_net.0.weight'].shape[1]}, 新: {Config.INPUT_DIM})")
                print("❌ 无法增量训练，请选择 [1. 全量训练] 重建模型！")
                return
                
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
            print("❌ 请重新选择 [1. 全量训练] 以适配新特征！")
            return
    else:
        print("[*] 初始化全新 Transformer 模型...")

    criterion = nn.HuberLoss() 
    
    # ==========================
    # Phase 1: 基础训练 (复习/重修)
    # ==========================
    print(f"\n>>> 阶段一：全市场扫描训练 (LR={LEARNING_RATE}, Epochs={EPOCHS})")
    
    # 获取股票列表 (根据模型类型)
    codes = DataProvider.get_stock_list(mode=model_type)
    
    all_X, all_y = [], []
    
    # 如果是增量训练，只随机抽查 30% 的股票进行“复习”，节省时间
    # 如果是全量训练，使用所有股票
    if mode == '2':
        # 激进模式下，股票池太大 (5000+)，优化采样策略:
        # 1. 基础采样率降至 10% (0.1)，保证广度但减少数量
        # 2. 硬上限 (Max Cap) 限制为 500 只，防止时间过长
        # 3. 动态过滤: 在下载数据后，如果发现成交量过低(僵尸股)或停牌，直接丢弃
        
        max_samples = 500
        sample_ratio = 0.1
        
        # 计算采样数量
        target_size = min(int(len(codes) * sample_ratio), max_samples)
        
        print(f"[*] 增量模式：随机抽取 {target_size} 只股票进行复习 (Pool: {len(codes)})...")
        training_codes = random.sample(codes, target_size)
    else:
        print(f"[*] 全量模式：使用全部 {len(codes)} 只股票进行训练...")
        training_codes = codes

    # 使用 DataLoader 并行加载数据
    # 增量模式只看最近 300 天，全量模式看最近 1000 天
    fetch_days = 300 if mode == '2' else Config.LOOKBACK + 500
    
    print(f"[*] 启动并行数据加载 (使用 4 个 worker)...")
    dataset = StockDataset(training_codes, train_mode=mode, fetch_days=fetch_days, lookback=Config.LOOKBACK)
    
    # Windows 下建议 num_workers 设为 0 以保证稳定性 (避免 WinError 10053 和内存溢出)
    # 虽然是单进程，但因为有增量缓存，速度依然很快
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    # 使用 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print(f"\n>>> 开始流式训练 (Stream Training)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        # 进度条显示当前 Epoch 进度
        pbar = tqdm(dataloader, total=len(training_codes), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for X_batch_list, y_batch_list in pbar:
            # X_batch_list 是一个列表，包含了一只股票的所有切片样本
            # 例如: Tensor shape [200, 60, 30]
            
            if len(X_batch_list) == 0: continue
            
            # 将列表转为 Tensor
            # 注意：这里只加载一只股票的数据到 GPU，内存占用极小
            X_stock = torch.tensor(np.array(X_batch_list), dtype=torch.float32).to(Config.DEVICE)
            y_stock = torch.tensor(np.array(y_batch_list), dtype=torch.float32).view(-1, 1).to(Config.DEVICE)
            
            # 在股票内部进行小批量训练 (Mini-batch within Stock)
            # 或者直接把整只股票作为一个 Batch 训练 (如果样本数不多，比如 200 个，完全可以)
            # 为了稳定，我们直接整只股票训练
            
            optimizer.zero_grad()
            out = model(X_stock)
            
            # === Loss 计算 ===
            base_loss = criterion(out, y_stock)
            
            # 获取输入序列的最后一天收盘价 (作为基准)
            close_idx = DataProcessor.FEATURE_COLS.index('close')
            last_close = X_stock[:, -1, close_idx].view(-1, 1)
            
            diff_pred = out - last_close
            diff_real = y_stock - last_close
            
            penalty = torch.where(diff_pred * diff_real < 0, base_loss * 2.0, torch.zeros_like(base_loss))
            loss = base_loss + penalty.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            current_loss = loss.item() * len(X_stock)
            total_loss += current_loss
            
            acc = ((diff_pred * diff_real) > 0).float().sum().item()
            total_acc += acc
            total_samples += len(X_stock)
            
            # 更新进度条后缀
            pbar.set_postfix({'Loss': f"{total_loss/total_samples:.6f}", 'Acc': f"{total_acc/total_samples*100:.2f}%"})

        # Epoch 结束打印
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_acc / total_samples * 100 if total_samples > 0 else 0
        print(f"Phase 1 | Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.2f}%")

    # ==========================
    # Phase 2: 错题本精调 (Feedback Loop)
    # ==========================
    print("\n>>> 阶段二：错题本精调 (Hard Example Mining)")
    bs.logout()  # 先断开旧连接
    import time
    time.sleep(1) # 休息一秒
    bs.login()   # 重新建立新连接
    fb_X, fb_y = get_feedback_data(model_type=model_type)
    
    if len(fb_X) > 0:
        X_fb = torch.tensor(np.array(fb_X), dtype=torch.float32).to(Config.DEVICE)
        y_fb = torch.tensor(np.array(fb_y), dtype=torch.float32).view(-1, 1).to(Config.DEVICE)
        
        print(f"[*] 针对 {len(X_fb)} 个严重错误样本进行特训...")
        
        # 特训使用极低的学习率，防止破坏已有知识
        # 增量模式下，对错题更敏感
        ft_lr = 0.0001 if mode == '1' else 0.0002
        optimizer_ft = optim.SGD(model.parameters(), lr=ft_lr, momentum=0.9)
        
        ft_epochs = 20 # 无论哪种模式，错题都要看 20 遍
        
        for epoch in range(ft_epochs):
            model.train()
            optimizer_ft.zero_grad()
            out = model(X_fb)
            loss = criterion(out, y_fb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer_ft.step()
            
            if (epoch+1) % 5 == 0:
                print(f"Fine-tune Epoch {epoch+1} | Loss: {loss.item():.6f}")
        print("✅ 错题吸收完毕！")
    else:
        print("[*] 暂无严重错题数据，跳过微调。")

    torch.save(model.state_dict(), model_path)
    print(f"\n{'='*50}")
    print(f"✅ {('全量' if mode=='1' else '增量')}训练完成！模型已保存至: {model_path}")
    print(f"{'='*50}\n")
    # bs.logout() # 注意：不要在这里 logout，否则会断开 auto_run 的连接

if __name__ == "__main__":
    try:
        train_model()
    finally:
        bs.logout()