import math
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 마스크된 self-attention 적용
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.layernorm1(tgt + self.dropout(tgt2))
        # 인코더와의 교차 어텐션 적용
        tgt2 = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.layernorm2(tgt + self.dropout(tgt2))
        # 피드포워드 네트워크 적용
        tgt2 = self.feed_forward(tgt)
        tgt = self.layernorm3(tgt + self.dropout(tgt2))
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.d_model = d_model
        self.linear_out = nn.Linear(d_model, output_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: [batch_size, tgt_seq_len]
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        output = self.linear_out(x)
        return output 