import torch.nn as nn
import torch
from src.config import Config
import torch.nn.functional as F
class CoherenceChecker(nn.Module):
    def __init__(self, level):
        super().__init__()
        embed_size = Config.vector_sizes[level+1]
        self.d1 = nn.Linear(embed_size, 4 * embed_size)
        self.d2 = nn.Linear(4 * embed_size, 4 * embed_size)
        self.out = nn.Linear(4 * embed_size, 1)
        self.out_prob = nn.Linear(4 * embed_size, 1)
        self.out.bias.data.fill_(-2.2) #better than random init
        self.out_prob.bias.data.fill_(-0.5) #better than random init
        #self.LayerNorm = nn.LayerNorm(4*embed_size)


        # for cnn forward
        self.out_classifier = nn.Linear(4 * embed_size, 3) #noise, encoded, reconstructed

        #for cnn forward
        self.num_filters = 32
        self.padding = Config.cnn_padding
        self.conv = nn.Conv1d(embed_size, self.num_filters, 1+2*self.padding,padding_mode='circular',padding=self.padding)  # <input_vector_size, num_filters, 1=unigram>
        self.max_pool = nn.MaxPool1d(Config.sequence_lengths[level+1])
        self.act = F.elu
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.d3 = nn.Linear(self.num_filters, 3)

    # TODO - Add Dropout
    def forward(self, x0):
        x = torch.tanh(self.d1(x0))
        #x = self.LayerNorm(x)
        x = torch.tanh(self.d2(x))
        #x = self.LayerNorm(x)
        scores = torch.sigmoid(self.out(x)) * Config.max_coherence_noise
        probs = torch.sigmoid(self.out_prob(x))
        class_predictions = torch.softmax(self.out_classifier(x),-1)
        return scores,probs,class_predictions

    def lower_forward(self, x0):
        x = torch.tanh(self.d1(x0))
        #x = self.LayerNorm(x)
        x = torch.tanh(self.d2(x))
        #x = self.LayerNorm(x)
        scores = torch.sigmoid(self.out(x)) * Config.max_coherence_noise
        probs = torch.sigmoid(self.out_prob(x))
        #class_predictions = torch.softmax(self.out_classifier(x),-1)

        #with CNN now
        x = torch.transpose(x0.unsqueeze(0), 1, 2) #todo: make better for the TPU no need for this reshaping
        x = self.conv(x)
        x = torch.transpose(x.squeeze(0),0,1)
        x = self.act(x)
        class_predictions = torch.softmax(self.d3(x),-1)
        return scores,probs,class_predictions




