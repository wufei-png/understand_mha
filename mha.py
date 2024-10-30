# 多头注意力的代码实现
import torch
# 导入库
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        # 查询、键和值的线性投影
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 输出线性投影
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
      batch_size, seq_length, d_model = x.size()
      print("x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)",x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2).shape)
      return x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, query, key, value, mask=None):

        # 线性投影
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        print("query shape:", query.shape)
        # 分割头部
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        print("scores shape:", scores.shape)
        # 如果提供了掩码，则应用掩码
        if mask is not None:
            scores += scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重并应用softmax
        attention_weights = torch.softmax(scores, dim=-1)
        print("attention_weights shape:", attention_weights.shape,"value shape:", value.shape)
        # 应用注意力到值
        attention_output = torch.matmul(attention_weights, value)
        print("attention_weattention_outputights shape:", attention_output.shape)
        # 合并头部
        batch_size, _, seq_length, d_k = attention_output.size()
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size,
        seq_length, self.d_model)

        # 线性投影
        attention_output = self.output_linear(attention_output)

        return attention_output

# 示例用法
d_model = 512
max_len = 100
num_heads = 8
d_ff = 2048

# 多头注意力
multihead_attn = MultiHeadAttention(d_model, num_heads)

# 示例输入序列
input_sequence = torch.randn(5, max_len, d_model)

# 多头注意力
attention_output= multihead_attn(input_sequence, input_sequence, input_sequence)
print("attention_output shape:", attention_output.shape)