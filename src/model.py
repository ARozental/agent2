from src.transformer import PositionalEncoding
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


class Model(nn.Module):
    def __init__(self, embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15, num_tokens=None):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_head, num_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.embed_size = embed_size

        #self.decoder = nn.Linear(embed_size, embed_size)
        self.out_transform = nn.Linear(embed_size, embed_size) #please don't call it decoder, transform is the name in bert and decoder is something else
        self.decoder_simple = nn.Linear(embed_size, num_tokens)

    def forward(self, src, mask):
        src = self.embedding(src)  # * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)  # Commenting out the positional encoder makes it learn faster => only for quick brown foxes

        output = self.transformer_encoder(src, mask)

        # # Using this makes it super fast and consistent (when using not MLM loss)
        # return F.softmax(self.decoder_simple(output), dim=1)

        output = self.out_transform(output)
        return torch.matmul(output, torch.transpose(self.embedding.weight, 0, 1).unsqueeze(0))  #[batch,seq_length,num_tokens]

    def decode(self, src, mask):
        logits = self.forward(src, mask)
        return torch.argmax(logits, dim=2)
