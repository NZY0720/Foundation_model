from encoder import TemporalEncoder, HierarchicalEncoder
from torch import nn
from torch.nn import functional as F

class PowerPM(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, num_relations):
        super(PowerPM, self).__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, model_dim, num_heads, num_layers)
        self.hierarchical_encoder = HierarchicalEncoder(model_dim, hidden_dim, num_relations)
        self.final_linear = nn.Linear(model_dim, 1)  # 映射到标量输出

    def forward(self, x, exogenous_vars=None, edge_index=None, edge_type=None):
        # 时间编码器
        temporal_out = self.temporal_encoder(x, exogenous_vars)

        # 层次编码器（如果没有图结构则跳过）
        if edge_index is not None and edge_type is not None:
            hierarchical_out = self.hierarchical_encoder(temporal_out, edge_index, edge_type)
        else:
            hierarchical_out = temporal_out

        # 全局特征池化
        global_feature = F.adaptive_avg_pool1d(hierarchical_out.permute(0, 2, 1), 1).squeeze(-1)

        # 通过线性层生成最终输出
        final_out = self.final_linear(global_feature)
        return final_out.squeeze(-1)

    def count_parameters(self):
        """
        计算模型参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
