from src.config import MODEL_CONFIG
from src.model import AgentModel
from src.simple_dataset import SimpleDataset
import torch

# Uncomment this to be word level only
# MODEL_CONFIG = MODEL_CONFIG[:1]

USE_CUDA = False
NUM_LEVELS = len(MODEL_CONFIG)

dataset = SimpleDataset(max_level=NUM_LEVELS)
device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
model = AgentModel(MODEL_CONFIG, num_tokens=dataset.num_tokens(), max_seq_length=dataset.tokenizer.max_lengths)
model.to(device)
model.train()

optimizer = torch.optim.Adagrad(model.parameters(), 0.01)

for epoch in range(500):
    print('Epoch', epoch + 1)

    for sentences, mask in dataset.iterator():
        model.train()
        # for sent in sentences:
        #     print('Original:', dataset.decode(sent))
        inputs = torch.tensor(sentences).to(device)
        mask = torch.BoolTensor(mask).to(device)
        optimizer.zero_grad()

        _, mlm_loss, coherence_loss, reconstruct_loss = model.fit(inputs, mask)
        (sum(mlm_loss) + sum(coherence_loss) + sum(reconstruct_loss)).backward()
        optimizer.step()

        model.eval()
        print('Word Level')
        if NUM_LEVELS == 1:
            inputs = inputs.reshape((inputs.size(0), 1, inputs.size(1)))
            mask = mask.reshape((mask.size(0), 1, mask.size(1)))
        for i, sent in enumerate(sentences):
            word_vec = model.encode(inputs[i], mask[i], level=0)
            pred = model.decode(word_vec, level=0)
            print(dataset.decode(pred), end='')
            if dataset.decode(pred) == dataset.decode(sent):
                print('   MATCHED!', end='')
            print('')

        if NUM_LEVELS > 1:
            print('Sentence Level')
            sent_vec = model.encode(inputs, mask)
            # sent_diff = sent_vec[0] - sent_vec[1]
            # print('Sent Diff Sum', torch.sum(sent_diff).item(), 'Norm', torch.norm(sent_diff).item())
            preds = model.decode(sent_vec, level=1)
            preds = preds.cpu().detach().numpy()
            for pred, sent in zip(preds, sentences):
                print(dataset.decode(pred), end='')
                if dataset.decode(pred) == dataset.decode(sent):
                    print('   MATCHED!', end='')
                print('')

            # Word Vector distance
            # word_vectors = [model.encode(inputs[i], mask[i], level=0) for i in range(len(sentences))]
            # word_vecs_from_sentences = model.decode(sent_vec, level=1, return_word_vectors=True)
            # for word_vecs, word_from_sent in zip(word_vectors, word_vecs_from_sentences):
            #     # print(word_vecs[:, :5])  # This shows the first 5 values of each word vector
            #     print('Distances:', torch.norm(word_vecs - word_from_sent, dim=1).detach().numpy())

    print('')
