import torch.nn.functional as F
import torch.nn as nn
import torch
from src.config import Config


def calc_reconstruction_loss(agent_level, vecotrs, mask, embeddings, labels):
    #matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]

    batch, seq_length = mask.shape
    decompressed = agent_level.decompressor(vecotrs)
    # print("d",decompressed.shape,mask.shape)
    post_decoder = agent_level.decoder(decompressed, mask) #[batch,seq_length,vec_size]




    # keep_positions = (torch.rand(batch, seq_length, 1)+1-0.5).floor() #1 => keep original 0, calc mlm,Config.mlm_rate
    # mlm_positions = 1-keep_positions
    # mask_positions = (torch.rand(batch, seq_length, 1)+0.8).floor() * mlm_positions #1 => replace with <mask>
    # special_mlm_positions = torch.rand(batch, seq_length, 1).floor() #1 => replace with original, 0 replace with random
    # random_replace_positions = mlm_positions*(1-mask_positions)*(1-special_mlm_positions)
    # replace_with_original_positions = mlm_positions*(1-mask_positions)*special_mlm_positions
    #
    # mask_vec_replacments = agent_level.mask_vector.repeat(batch*seq_length).view(batch,seq_length,vec_size)
    # random_indexes = torch.fmod(torch.randperm(batch* seq_length), embeddings.shape[0]) #todo: make sure the pad token is not here, also no join for levels 0 and 1
    # random_vec_replacments = torch.index_select(embeddings, 0,random_indexes).view(batch, seq_length, vec_size)
    #
    # pre_encoder = keep_positions*matrices + mask_positions*mask_vec_replacments + random_replace_positions*random_vec_replacments + replace_with_original_positions*matrices
    # post_encoder = agent_level.encoder(pre_encoder, mask)


    logits = torch.matmul(post_decoder,torch.transpose(embeddings,0,1)) #[batch,max_length,embedding_size)
    reconstruction_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels
        ,ignore_index=Config.pad_token_id
        ,reduction='none' #gives mlm loss from each of [batch,words]
    ).mean(-1)

    return reconstruction_losses


class ReconstructionLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Model is the current AgentLevel
        self.pad_token_id = 0

    def forward(self, inputs, mask):
        vector = self.model.encode(inputs, mask)  # This calls the encoder and the compressor
        decompressed = self.model.decompressor(vector)
        output = self.model.decoder(tgt=decompressed, memory=decompressed)
        emb_weight = torch.transpose(self.model.encoder.embedding.weight, 0, 1).unsqueeze(0)
        logits = torch.matmul(output, emb_weight)

        return F.cross_entropy(
            logits.transpose(1, 2),
            inputs,
            ignore_index=self.pad_token_id
        )
