from data_processor import DataProcessor, EnergyDataset
from torch.utils.data import DataLoader
from PM_Model import PowerPM
from torch import nn, optim
import torch
from tqdm import tqdm  # 引入 tqdm 进度条库

def main():
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    processor = DataProcessor(file_path="daily_dataset.csv", seq_length=7)
    data = processor.load_data()
    sequences, labels = processor.create_sequences(data)

    # 创建数据集和数据加载器
    dataset = EnergyDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型
    model = PowerPM(
        input_dim=6,  # 输入特征数量
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并显示进度条
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 创建 tqdm 进度条
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x, y = x.to(device), y.to(device)

                # 创建与 x 对应的外生变量（示例为全零张量）
                exogenous_vars = torch.zeros_like(x, device=device)

                optimizer.zero_grad()
                output = model(x, exogenous_vars, None, None)  # 确保 exogenous_vars 不为 None
                loss = criterion(output, y)  # 输出和目标匹配
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # 更新进度条
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)

        print(f"Epoch {epoch + 1} completed, Average Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    main()
