from data_processor import BlockDataProcessor, EnergyDataset
from PM_Model import PowerPM
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from tqdm import tqdm
import config  # 导入配置文件


def save_model(model, path):
    """
    保存模型权重到指定路径
    :param model: 训练后的模型
    :param path: 保存路径
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    从指定路径加载模型权重
    :param model: 模型实例
    :param path: 模型权重路径
    :param device: 加载设备 (CPU/GPU)
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")


def main():
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化数据处理器
    processor = BlockDataProcessor(
        data_dir=config.DATA_DIR,
        household_info_path=config.HOUSEHOLD_INFO_PATH,
        weather_data_path=config.WEATHER_DATA_PATH,
        seq_length=config.SEQ_LENGTH
    )

    # 加载并预处理数据
    raw_block_data = processor.load_all_blocks()
    household_info = processor.load_household_info()
    weather_data = processor.load_weather_data()
    time_series_data = processor.preprocess_data(raw_block_data, household_info, weather_data)
    sequences, labels = processor.create_sequences(time_series_data)

    # 创建数据集和数据加载器
    dataset = EnergyDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 初始化模型
    input_dim = sequences.shape[2]  # 输入特征维度
    model = PowerPM(
        input_dim=input_dim,  # 动态调整输入特征数量
        model_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        num_relations=1  # 无关系图时设置为1
    ).to(device)  # 模型移动到 GPU

    # 输出模型参数量
    print(f"Model Parameter Count: {model.count_parameters()}")

    # 损失函数和优化器
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练模型
    best_loss = float("inf")  # 用于保存最佳模型
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x, y = x.to(device), y.to(device)

                # 数据预处理：检查并修复异常值
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

                optimizer.zero_grad()
                output = model(x, None, None, None)  # 确保 exogenous_vars 不为 None

                # 输出裁剪，防止溢出
                output = torch.clamp(output, min=-1e6, max=1e6)

                loss = criterion(output, y)

                # 检查损失值是否异常
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf detected in loss at batch {batch_idx}. Skipping batch.")
                    continue

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        # 更新学习率调度器
        scheduler.step(total_loss / len(dataloader))

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, config.MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
