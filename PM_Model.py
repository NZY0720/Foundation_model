from encoder import TemporalEncoder, HierarchicalEncoder
from torch import nn

class PowerPM(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, num_relations):
        super(PowerPM, self).__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, model_dim, num_heads, num_layers)
        self.hierarchical_encoder = HierarchicalEncoder(model_dim, hidden_dim, num_relations)
        self.final_linear = nn.Linear(model_dim, 1)  # 映射到标量输出

    def forward(self, x, exogenous_vars=None, edge_index=None, edge_type=None):
        # 时间编码器
        temporal_out = self.temporal_encoder(x, exogenous_vars)  # 输出形状 [batch_size, seq_length, model_dim]

        # 层次编码器（如果没有图结构则跳过）
        if edge_index is not None and edge_type is not None:
            hierarchical_out = self.hierarchical_encoder(temporal_out, edge_index, edge_type)
        else:
            hierarchical_out = temporal_out

        # 对最后一个时间步的输出应用线性变换
        final_out = self.final_linear(hierarchical_out[:, -1, :])  # 取最后一个时间步
        return final_out.squeeze(-1)  # 输出形状 [batch_size]

    def count_parameters(self):
        """
        计算模型参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
