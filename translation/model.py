import math

import torch
import torch.nn as nn

PAD_ID = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerMT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x == PAD_ID  # (batch, seq) True where padding

    def encode(self, src: torch.Tensor):
        src_pad_mask = self.make_pad_mask(src)
        src_emb = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return memory, src_pad_mask

    def decode_step(self, tgt: torch.Tensor, memory: torch.Tensor, src_pad_mask: torch.Tensor):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_emb = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(out)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src)
        return self.decode_step(tgt_in, memory, src_pad_mask)
