import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value, mask=None):
        # query, key, value의 shape: [batch_size, seq_len, d_model]
        batch_size = query.size(0)

        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        # 멀티헤드로 변환: [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = Q / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        x = torch.matmul(attn, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(x)
        return output 