import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, src_input_dim, tgt_output_dim, d_model=512, num_encoder_layers=6,
                 num_decoder_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_input_dim, d_model, num_encoder_layers, num_heads, d_ff, dropout, max_len)
        self.decoder = TransformerDecoder(tgt_output_dim, d_model, num_decoder_layers, num_heads, d_ff, dropout, max_len)

    def generate_square_subsequent_mask(self, sz):
        # 디코더 마스킹용 삼각 행렬 생성
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output 