import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class BlockDataProcessor:
    def __init__(self, data_dir, household_info_path, weather_data_path, seq_length=48):
        self.data_dir = data_dir
        self.household_info_path = household_info_path
        self.weather_data_path = weather_data_path
        self.seq_length = seq_length

    def load_all_blocks(self):
        """
        读取所有 block 文件并合并。
        :return: pandas.DataFrame
        """
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.startswith("block_")]
        data_frames = []
        for file in all_files:
            df = pd.read_csv(file)
            data_frames.append(df)
        data = pd.concat(data_frames, ignore_index=True)
        return data

    def load_household_info(self):
        """
        加载 household 辅助信息。
        :return: pandas.DataFrame
        """
        return pd.read_csv(self.household_info_path)

    def load_weather_data(self):
        """
        加载天气数据并进行时间格式转换。
        :return: pandas.DataFrame
        """
        weather_data = pd.read_csv(self.weather_data_path)
        weather_data["time"] = pd.to_datetime(weather_data["time"])
        return weather_data

    def preprocess_data(self, block_data, household_info, weather_data):
        """
        将 block 数据与 household 和天气数据合并，生成时间序列数据。
        :param block_data: pandas.DataFrame 原始 block 数据
        :param household_info: pandas.DataFrame household 数据
        :param weather_data: pandas.DataFrame 天气数据
        :return: pandas.DataFrame 合并后的时间序列数据
        """
        # 合并 block 数据和 household 辅助信息
        merged_data = pd.merge(block_data, household_info, on="LCLid", how="left")
        
        # 展开时间序列并加入天气数据
        time_series = []
        for _, row in merged_data.iterrows():
            for i in range(48):
                timestamp_str = f"{row['day']} {i//2:02d}:{(i%2)*30:02d}"
                
                # 转换为时间戳
                try:
                    timestamp = pd.to_datetime(timestamp_str)  # 转换为时间戳
                except ValueError as e:
                    print(f"Error parsing timestamp: {timestamp_str} -> {e}")
                    continue  # 跳过无法解析的时间戳

                # 合并天气数据
                weather_row = weather_data[weather_data["time"] == timestamp]
                if not weather_row.empty:
                    weather_features = weather_row.iloc[0].to_dict()  # 获取天气特征
                else:
                    weather_features = {col: np.nan for col in weather_data.columns}  # 缺失填充
                
                time_series.append({
                    "LCLid": row["LCLid"],
                    "timestamp": timestamp,
                    "value": row[f"hh_{i}"],
                    "stdorToU": row["stdorToU"],
                    "Acorn": row["Acorn"],
                    "Acorn_grouped": row["Acorn_grouped"],
                    **weather_features
                })
        
        time_series_df = pd.DataFrame(time_series)
        
        # 填补缺失值
        time_series_df.fillna(method="ffill", inplace=True)
        time_series_df.fillna(method="bfill", inplace=True)
        return time_series_df

    def create_sequences(self, data):
        """
        根据时间序列数据生成训练样本。
        :param data: pandas.DataFrame 时间序列数据
        :return: torch.Tensor 特征和目标
        """
        sequences = []
        labels = []
        grouped = data.groupby("LCLid")  # 按用户分组
        for _, group in grouped:
            group = group.sort_values("timestamp")  # 按时间排序
            values = group["value"].values
            # 对类别特征进行 One-Hot 编码
            stdorToU = pd.get_dummies(group["stdorToU"], drop_first=True).values
            acorn_grouped = pd.get_dummies(group["Acorn_grouped"], drop_first=True).values
            precip_type = pd.get_dummies(group["precipType"], drop_first=True).values
            icon = pd.get_dummies(group["icon"], drop_first=True).values
            
            # 选取数值型天气特征
            weather_features = group[["temperature", "humidity", "pressure", "windSpeed"]].values

            auxiliary_features = np.hstack([
                stdorToU, acorn_grouped, precip_type, icon, weather_features
            ])  # 辅助特征
            for i in range(len(values) - self.seq_length):
                main_sequence = values[i:i + self.seq_length]
                aux_features = auxiliary_features[i:i + self.seq_length]
                sequences.append(np.hstack([main_sequence.reshape(-1, 1), aux_features]))  # 合并主序列和辅助特征
                labels.append(values[i + self.seq_length])  # 第 seq_length + 1 时间步的目标值
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return torch.tensor(sequences), torch.tensor(labels)


class EnergyDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        自定义数据集，用于包装时间序列数据。
        :param sequences: 输入特征 (torch.Tensor)
        :param labels: 目标值 (torch.Tensor)
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
