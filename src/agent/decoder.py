from src.transformer import PositionalEncoding,EncoderLayer,TransformerEncoder
import torch.nn as nn
from src.config import Config

class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate) #should it be always 0?
        #self.decoder_layers = nn.TransformerDecoderLayer(Config.vector_sizes[level], Config.num_heads[level], Config.vector_sizes[level], Config.drop_rate, activation="relu") #change to swiglu
        #self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, Config.num_transformer_layers[level])
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level],Config.vector_sizes[level], Config.drop_rate,activation="gelu") # change to swiglu
        self.transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])
        self.embed_size = Config.vector_sizes[level]

    def forward(self, src, mask, eos_positions):
        src = src.transpose(0, 1)
        eos_positions = eos_positions.transpose(0, 1).unsqueeze(-1) #todo: fix?? due to the positional encoding not all eos are the same vec, #do we even need pos encoding here?

        eos_value = eos_positions*src
        src = src-self.pos_encoder(src) # * math.sqrt(Config.vector_sizes[level])
        src = eos_positions*eos_value+(1-eos_positions)*src

        return self.transformer_encoder(src, src_key_padding_mask=mask,eos_positions=eos_positions).transpose(0, 1)




        # tgt = tgt.transpose(0, 1)
        # memory = memory.transpose(0, 1)
        # if len(tgt.size()) == 2:
        #     tgt = self.embedding(tgt)  # * math.sqrt(self.embed_size)
        # memory = self.pos_encoder(memory)
        #
        # output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask)
        # return output.transpose(0, 1)
