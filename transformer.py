import math
import torch
import torch.nn as nn

# 위치 인코딩 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 멀티헤드 어텐션 (Multi-Head Attention)
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
        
        # 선형 사영
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        
        # 멀티헤드로 변환: [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 스케일링된 쿼리
        Q = Q / math.sqrt(self.d_k)
        
        # 어텐션 점수 계산: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 어텐션을 적용한 결과: [batch_size, num_heads, seq_len, d_k]
        x = torch.matmul(attn, V)
        
        # 헤드 결합: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(x)
        return output


# 피드포워드 네트워크 (Feed Forward Network)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# Transformer 인코더 레이어
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # 멀티헤드 self-attention + residual connection과 layer normalization 적용
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        x = self.layernorm1(src + self.dropout(attn_output))
        
        # 피드포워드 네트워크 + residual connection과 layer normalization 적용
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x


# Transformer 인코더
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, dropout=0.1, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.d_model = d_model
        
    def forward(self, src, src_mask=None):
        # src: [batch_size, seq_len]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# Transformer 디코더 레이어
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
        
        # 인코더의 출력과의 교차 어텐션 적용
        tgt2 = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.layernorm2(tgt + self.dropout(tgt2))
        
        # 피드포워드 네트워크 적용
        tgt2 = self.feed_forward(tgt)
        tgt = self.layernorm3(tgt + self.dropout(tgt2))
        return tgt


# Transformer 디코더
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


# 전체 Transformer 모델 (Encoder-Decoder 구조)
class Transformer(nn.Module):
    def __init__(self, src_input_dim, tgt_output_dim, d_model=512, num_encoder_layers=6, num_decoder_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_input_dim, d_model, num_encoder_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = TransformerDecoder(tgt_output_dim, d_model, num_decoder_layers, num_heads, d_ff, dropout, max_len)
        
    def generate_square_subsequent_mask(self, sz):
        # 디코더의 마스킹용 삼각 행렬 생성
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output


if __name__ == '__main__':
    # 간단한 테스트 예시
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 10
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = Transformer(src_vocab_size, tgt_vocab_size)
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 디코더를 위한 마스크 생성 (아래 코드는 예시입니다.)
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    
    output = model(src, tgt, tgt_mask=tgt_mask)
    print("Transformer 출력 모양:", output.shape) 