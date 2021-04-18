from src.transformer import PositionalEncoding,EncoderLayer, TransformerEncoder
import torch.nn as nn
import torch
from src.config import Config
import math
import torch.nn.functional as F
from src.utils import attention




class Pndb(nn.Module):
    def __init__(self,level=1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level],Config.vector_sizes[level], Config.drop_rate,activation="gelu") # change to swiglu
        self.pndb_transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level]) #not sure we need it...

        self.num_questions = Config.pndb_questions
        self.questions = nn.Parameter(torch.rand([self.num_questions, Config.vector_sizes[level]], requires_grad=True)) #global Q matrix
        self.questions2 = nn.Parameter(torch.rand([self.num_questions, Config.vector_sizes[level]], requires_grad=True)) #global Q matrix

        self.to_k = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level])
        self.to_k2 = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level])
        #self.to_v = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level]) #should it be the identity matrix??
        self.to_v2 = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level]) #should it be the identity matrix??
        self.ignore1 = nn.Linear(Config.vector_sizes[level], 1)
        self.ignore2 = nn.Linear(Config.vector_sizes[level], 1)

        self.update11 = nn.Linear(Config.vector_sizes[level], 1)
        self.update12 = nn.Linear(Config.vector_sizes[level], 1)

        self.update21 = nn.Linear(Config.vector_sizes[level], 1)
        self.update22 = nn.Linear(Config.vector_sizes[level], 1)
        self.b1 = nn.Parameter(torch.rand(1, requires_grad=True))


        self.to_output_k = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level])

    def ignore_gate(self,x,g):
      return torch.sigmoid(g(x))

    def update_gate(self,x,A,g1,g2,b):
      return torch.sigmoid(g1(x)+g2(A)+b)

    def create_A_matrix(self,raw_embedding_matrices,mask):
        #input is the matrices from a single book only! each node should have a root_id so we can make sure of that
        k = self.to_k(raw_embedding_matrices)
        v = raw_embedding_matrices*self.ignore_gate(raw_embedding_matrices,self.ignore1)
        A = attention(self.questions,k,v,Config.vector_sizes[1],mask=mask) #[batch,num_questions,hidden]
        A = A.mean(0) #we can have a sum here and subtract later
        return A


    def create_A2_matrix(self,post_encoder_matrices,mask):
        #input is the matrices from a single book only! each node should have a root_id so we can make sure of that
        k = self.to_k2(post_encoder_matrices)
        v = self.to_v2(post_encoder_matrices)*self.ignore_gate(post_encoder_matrices,self.ignore2)
        A = attention(self.questions2,k,v,Config.vector_sizes[1],mask=mask) #[batch,num_questions,hidden]
        A = A.mean(0)
        return A

    def get_data_from_A_matrix(self,A,post_decoder_matrices):
      k = self.to_output_k(post_decoder_matrices)
      A2 = attention(k, self.questions, A, self.num_questions)  # [batch,seq_length,hidden]
      gate_values = self.update_gate(post_decoder_matrices, A2, self.update11, self.update12, self.b1)
      return post_decoder_matrices + A2*gate_values