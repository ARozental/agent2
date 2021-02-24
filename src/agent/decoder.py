from src.transformer import PositionalEncoding
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embedding, embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15):
        super().__init__()
        self.embedding = embedding
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        decoder_layers = nn.TransformerDecoderLayer(embed_size, num_head, num_hidden, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.embed_size = embed_size

    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        if len(tgt.size()) == 2:
            tgt = self.embedding(tgt)  # * math.sqrt(self.embed_size)
        memory = self.pos_encoder(memory)

        output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return output.transpose(0, 1)
