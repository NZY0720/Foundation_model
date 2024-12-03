# 配置文件：config.py

# 数据路径配置
DATA_DIR = "data"  # 修改为实际 block 文件所在路径
HOUSEHOLD_INFO_PATH = "data/informations_households.csv"  # household 信息路径
WEATHER_DATA_PATH = "data/weather_hourly_darksky.csv"  # 天气数据路径

# 模型保存路径
MODEL_SAVE_PATH = "trained_model.pth"

# 其他配置
SEQ_LENGTH = 48  # 时间序列长度
BATCH_SIZE = 32  # 批量大小
NUM_EPOCHS = 10  # 训练轮次
LEARNING_RATE = 0.0001  # 学习率
