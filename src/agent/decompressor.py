import torch.nn as nn
import torch
from src.config import Config

class Decompressor(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level

        self.recurrent = nn.LSTM(Config.vector_sizes[level+1], Config.vector_sizes[level+1], dropout=Config.drop_rate, batch_first=True)
        self.out_projection = nn.Linear(Config.vector_sizes[level+1], Config.vector_sizes[level]) #use ,proj_size instead
        self.dropout = nn.Dropout(Config.drop_rate)

    def forward(self, x):
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # h0.size=c0.size = (num_layers * num_directions, batch, hidden_size)        #input: seq_len, batch, input_size
        state_h, state_c = (torch.zeros(1, 1, Config.vector_sizes[self.level+1]), torch.zeros(1, 1, Config.vector_sizes[self.level+1]))
        #todo: state_h, state_c should probably be trainable params
        seq = []
        last_input = x.unsqueeze(0)
        for i in range(Config.sequence_lengths[self.level]): #todo? there should probably be a pure torch way to do it
            output, (state_h, state_c) = self.recurrent(last_input, (state_h, state_c))
            seq.append(output)
            last_input = output

        seq = torch.cat(seq,0).transpose(0, 1) #[batch,max_length,top_text_vec_size]
        seq = self.out_projection(self.dropout(seq))
        return seq #[batch,max_length,vec_size]

    #closest vec / first close vec / have the last embedding matrix and choose first stop / other option
        ####not really children vecs as there is no decoder here!!!
    # def vec_to_children_vecs_old(self,x,embedding_matrix):
    #     #0th-element is the eos token; X is a vector
    #
    #     seq = self.forward(x.unsqueeze(0)).squeeze()
    #     output = torch.matmul(seq, torch.transpose(embedding_matrix, 0, 1))
    #     output = torch.argmax(output, dim=1).tolist() #selected vector_id for each position, first 0 is eos
    #     children_vecs = [seq[0]]
    #
    #     for i in range(1,len(output)):
    #         if output[i]==0:
    #             break
    #         children_vecs.append(seq[i])
    #
    #     return children_vecs
