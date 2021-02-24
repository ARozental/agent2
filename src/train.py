from src.losses import MLMLoss, CoherenceLoss, ReconstructionLoss
from src.config import MODEL_CONFIG
from src.model import Model
from src.simple_dataset import SimpleDataset
import torch

USE_CUDA = False

dataset = SimpleDataset(max_level=len(MODEL_CONFIG))

device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

NUM_TOKENS = len(dataset.tokenizer.tokenizer)

model = Model(MODEL_CONFIG, num_tokens=NUM_TOKENS, max_seq_length=dataset.tokenizer.max_lengths)#dataset.max_length)
model.to(device)
model.train()

optimizer = torch.optim.Adagrad(model.parameters(), 0.01)

for epoch in range(500):
    print('Epoch', epoch + 1)

    for sentences, mask in dataset.iterator():
        for sent in sentences:
            print('Original:', dataset.decode(sent))
        inputs = torch.tensor(sentences).to(device)
        mask = torch.BoolTensor(mask).to(device)
        optimizer.zero_grad()

        _, mlm_loss, coherence_loss, reconstruct_loss = model.fit(inputs, mask)
        (sum(mlm_loss) + sum(coherence_loss) + sum(reconstruct_loss)).backward()
        optimizer.step()

        with torch.no_grad():
            preds = model.levels[0].decode(inputs, mask)
        preds = preds.cpu().detach().numpy()
        for pred, sent in zip(preds, sentences):
            print('Pred:', dataset.decode(pred), end='')
            if dataset.decode(pred) == dataset.decode(sent):
                print('   MATCHED!', end='')
            print('')

    print('')
