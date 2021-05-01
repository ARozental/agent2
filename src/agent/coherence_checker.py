import torch.nn as nn
import torch
from src.config import Config

class CoherenceChecker(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.d1 = nn.Linear(embed_size, 4 * embed_size)
        self.d2 = nn.Linear(4 * embed_size, 4 * embed_size)
        self.out = nn.Linear(4 * embed_size, 1)
        self.out_prob = nn.Linear(4 * embed_size, 1)
        self.out.bias.data.fill_(-2.2) #better than random init
        self.out_prob.bias.data.fill_(-0.5) #better than random init
        #self.LayerNorm = nn.LayerNorm(4*embed_size)

        self.out_classifier = nn.Linear(4 * embed_size, 3) #noise, encoded, reconstructed



    # TODO - Add Dropout
    def forward(self, x):
        x = torch.tanh(self.d1(x))
        #x = self.LayerNorm(x)
        x = torch.tanh(self.d2(x))
        #x = self.LayerNorm(x)
        scores = torch.sigmoid(self.out(x)) * Config.max_coherence_noise
        probs = torch.sigmoid(self.out_prob(x))
        class_predictions = torch.softmax(self.out_classifier(x),-1)
        return scores,probs,class_predictions




