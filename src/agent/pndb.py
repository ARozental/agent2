from src.transformer import PositionalEncoding, EncoderLayer, TransformerEncoder
import torch.nn as nn
import torch
from src.config import Config
import math
import torch.nn.functional as F
from src.utils import attention, prob_to_logit
from src.transformer import Rotary


class Pndb(nn.Module):
    def __init__(self, level=1):
        super().__init__()
        #self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)
        self.rotary = Rotary(Config.sequence_lengths[level], Config.vector_sizes[level])

        encoder_layers = EncoderLayer(Config.vector_sizes[level], 1, Config.vector_sizes[level],
                                      Config.drop_rate, activation="gelu",rotary=self.rotary)  # change to swiglu
        self.pndb_transformer_encoder_write = TransformerEncoder(encoder_layers, 2)  # not sure we need it...
        self.pndb_transformer_encoder_read = TransformerEncoder(encoder_layers, 2)  # not sure we need it...

        if Config.use_pndb1 is not None:
            self.questions = nn.Parameter(
                torch.rand([Config.use_pndb1, Config.vector_sizes[level]], requires_grad=True))  # global Q matrix
            self.to_k = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level],bias=False)
            # self.to_v = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level]) #should it be the identity matrix??
            self.ignore1 = nn.Linear(Config.vector_sizes[level], 1)
            self.update11 = nn.Linear(Config.vector_sizes[level], 1)
            self.update12 = nn.Linear(Config.vector_sizes[level], 1)
            self.b1 = nn.Parameter(torch.rand(1, requires_grad=True))
            self.to_output_k = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level])

    def ignore_gate(self, x, g):
        return torch.sigmoid(g(x))

    def update_gate(self, x, A, g1, g2, b):
        return torch.sigmoid(g1(x) + g2(A) + b - 4.0) #todo: move to init b

    def create_A_matrix(self, raw_embedding_matrices, real_positions):
        # input is the matrices from a single book only! each node should have a root_id so we can make sure of that
        k = self.to_k(raw_embedding_matrices)

        #for_ignore = self.pndb_transformer_encoder_write(raw_embedding_matrices.transpose(0, 1), src_key_padding_mask=torch.log(real_positions)).transpose(0, 1)
        for_ignore = self.pndb_transformer_encoder_write(raw_embedding_matrices, src_key_padding_mask=torch.log(real_positions))
        v = raw_embedding_matrices * self.ignore_gate(for_ignore, self.ignore1)
        A = attention(self.questions, k, v, Config.vector_sizes[1], real_positions=real_positions)  # [batch,num_questions,hidden]
        A = A.mean(0)  # we can have a sum here and subtract later
        return A

    def old_get_data_from_A_matrix(self, A, post_decoder_matrices):
        k = self.to_output_k(post_decoder_matrices)
        A2 = attention(k, self.questions, A, Config.use_pndb1)  # [batch,seq_length,hidden]
        gate_values = self.update_gate(post_decoder_matrices, A2, self.update11, self.update12, self.b1)
        return post_decoder_matrices + A2 * gate_values

    def get_data_from_A_matrix(self, A1s,pndb_lookup_ids, post_decoder_matrices,real_positions_for_mask):
        #for_update = self.pndb_transformer_encoder_read(post_decoder_matrices.transpose(0, 1), src_key_padding_mask=torch.log(real_positions_for_mask)).transpose(0, 1)
        for_update = self.pndb_transformer_encoder_read(post_decoder_matrices, src_key_padding_mask=prob_to_logit(real_positions_for_mask))

        k = self.to_output_k(post_decoder_matrices)
        selected_A1s = torch.index_select(A1s, 0, pndb_lookup_ids) #todo: is this a horrible place where the memory explodes?
        A2 = attention(k, self.questions, selected_A1s, Config.use_pndb1)  # [batch,seq_length,hidden]
        gate_values = self.update_gate(for_update, A2, self.update11, self.update12, self.b1)
        return (post_decoder_matrices * (1 - gate_values)) + (A2 * gate_values), gate_values


