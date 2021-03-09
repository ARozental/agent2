from src.transformer import PositionalEncoding
import torch.nn as nn
from src.config import Config

class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate) #should it be always 0?
        self.decoder_layers = nn.TransformerDecoderLayer(Config.vector_sizes[level], Config.num_heads[level], Config.vector_sizes[level], Config.drop_rate, activation="relu") #change to swiglu
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, Config.num_transformer_layers[level])
        self.embed_size = Config.vector_sizes[level]

    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        if len(tgt.size()) == 2:
            tgt = self.embedding(tgt)  # * math.sqrt(self.embed_size)
        memory = self.pos_encoder(memory)

        output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return output.transpose(0, 1)
