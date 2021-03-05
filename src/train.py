from src.config import MODEL_CONFIG
from src.model import AgentModel
from src.simple_dataset import SimpleDataset
import torch

# MODEL_CONFIG = MODEL_CONFIG[:1]  # Uncomment this to be word level only
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
    print(torch.__version__)

    for batch in dataset.iterator():
        model.train()
        optimizer.zero_grad()

        _, mlm_loss, coherence_loss, reconstruct_loss = model.fit(batch)
        (sum(mlm_loss) + sum(coherence_loss) + sum(reconstruct_loss)).backward()
        optimizer.step()

        model.eval()
        print('Word Level')
        examples = dataset.debug_examples(level=0)
        word_vecs = model.encode(examples)
        preds = model.debug_decode(word_vecs, level=0)
        for pred, word in zip(preds, examples):
            print(dataset.decode(pred), end='')
            if dataset.decode(pred) == dataset.decode(word):
                print('   MATCHED!', end='')
            print('')

        if NUM_LEVELS > 1:
            print('Sentence Level')
            examples = dataset.debug_examples(level=1)
            sent_vec = model.encode(examples)
            preds = model.debug_decode(sent_vec, level=1)
            preds = preds.cpu().detach().numpy()
            for pred, sent in zip(preds, examples):
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
