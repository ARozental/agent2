from src.config import Config
from src.model import AgentModel
from src.simple_dataset import SimpleDataset
from src.tree_dataset import TreeDataset
from src.pre_processing import TreeTokenizer
import torch
import torch.nn as nn

# MODEL_CONFIG = MODEL_CONFIG[:1]  # Uncomment this to be word level only
USE_CUDA = False
#config = Config()

dataset = TreeDataset()
device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
model = AgentModel(TreeTokenizer())
model.to(device)
# model.train()
model.eval()
for batch_tree in dataset.iterator():
    model.forward(batch_tree)
    break


optimizer = torch.optim.Adam(model.parameters(), 0.001)

for epoch in range(2001):
    #print('Epoch', epoch + 1)

    for batch in dataset.iterator():
        model.train()
        optimizer.zero_grad()

        loss_object,total_loss = model.forward(batch)
        print(epoch, total_loss.item())
        if epoch%100==0:
            print(loss_object)
        total_loss.backward()
        optimizer.step()
#
#         model.eval()
#         print('Word Level')
#         examples = dataset.debug_examples(level=0)
#         word_vecs = model.encode(examples)
#         preds = model.debug_decode(word_vecs, level=0)
#         for pred, word in zip(preds, examples):
#             print(dataset.decode(pred), end='')
#             if dataset.decode(pred) == dataset.decode(word):
#                 print('   MATCHED!', end='')
#             print('')
#
#         if NUM_LEVELS > 1:
#             print('Sentence Level')
#             examples = dataset.debug_examples(level=1)
#             sent_vec = model.encode(examples)
#             preds = model.debug_decode(sent_vec, level=1)
#             preds = preds.cpu().detach().numpy()
#             for pred, sent in zip(preds, examples):
#                 print(dataset.decode(pred), end='')
#                 if dataset.decode(pred) == dataset.decode(sent):
#                     print('   MATCHED!', end='')
#                 print('')
#
#             # Word Vector distance
#             # word_vectors = [model.encode(inputs[i], mask[i], level=0) for i in range(len(sentences))]
#             # word_vecs_from_sentences = model.decode(sent_vec, level=1, return_word_vectors=True)
#             # for word_vecs, word_from_sent in zip(word_vectors, word_vecs_from_sentences):
#             #     # print(word_vecs[:, :5])  # This shows the first 5 values of each word vector
#             #     print('Distances:', torch.norm(word_vecs - word_from_sent, dim=1).detach().numpy())
#
#     print('')
