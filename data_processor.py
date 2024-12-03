import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DataProcessor:
    def __init__(self, file_path, seq_length=7):
        """
        数据处理器，用于加载和预处理数据，构造时间序列输入。
        :param file_path: 数据文件路径 (CSV)
        :param seq_length: 时间序列窗口长度
        """
        self.file_path = file_path
        self.seq_length = seq_length

    def load_data(self):
        """
        加载 CSV 数据文件并解析日期列。
        :return: pandas.DataFrame
        """
        data = pd.read_csv(self.file_path, parse_dates=["day"])
        return data

    def create_sequences(self, data):
        """
        将数据转换为时间序列格式，按用户分组构造滑动窗口。
        :param data: pandas.DataFrame 原始数据
        :return: (torch.Tensor, torch.Tensor) 输入序列和对应目标值
        """
        sequences = []
        labels = []

        # 按 LCLid 分组
        grouped = data.groupby("LCLid")
        for _, group in grouped:
            group = group.sort_values("day")  # 按日期排序
            # 提取特征列
            features = group[[
                "energy_median", "energy_mean", "energy_max", "energy_count", "energy_std", "energy_min"
            ]].values
            # 提取目标列
            targets = group["energy_sum"].values

            # 滑动窗口构造序列
            for i in range(len(group) - self.seq_length):
                sequences.append(features[i:i + self.seq_length])
                labels.append(targets[i + self.seq_length])

        # 转换为 numpy 数组再转换为 Tensor
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        return torch.tensor(sequences), torch.tensor(labels)


class EnergyDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        自定义数据集，用于包装时间序列数据和目标值。
        :param sequences: 输入序列 (torch.Tensor)
        :param labels: 目标值 (torch.Tensor)
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        """
        数据集大小
        :return: int
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 索引
        :return: (torch.Tensor, torch.Tensor) 输入序列和目标值
        """
        return self.sequences[idx], self.labels[idx]
