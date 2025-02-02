import torch
from transformer.model import Transformer

def main():
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 10
    src_vocab_size = 1000
    tgt_vocab_size = 1000

    model = Transformer(src_vocab_size, tgt_vocab_size)
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 디코더를 위한 마스크 생성
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    output = model(src, tgt, tgt_mask=tgt_mask)
    print("Transformer 출력 모양:", output.shape)

if __name__ == '__main__':
    main() 