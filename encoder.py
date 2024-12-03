from torch import nn
from torch_geometric.nn import RGCNConv

# 时间编码器
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TemporalEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, model_dim)  # 将输入映射到 model_dim
        self.exogenous_linear = nn.Linear(input_dim, model_dim)  # 映射外生变量

        # 创建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,  # Feedforward 层大小
            batch_first=True  # 保持输入的 batch 维度在第一维
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, exogenous_vars=None):
        # 将输入映射到模型维度
        x = self.linear(x)

        # 如果外生变量不为空，则映射并相加
        if exogenous_vars is not None:
            exogenous_vars = self.exogenous_linear(exogenous_vars)
            x = x + exogenous_vars

        # 使用 Transformer 编码器
        x = self.transformer_encoder(x)
        return x

# 层次编码器
class HierarchicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations):
        super(HierarchicalEncoder, self).__init__()
        self.rgcn = RGCNConv(input_dim, hidden_dim, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn(x, edge_index, edge_type)
        return x
