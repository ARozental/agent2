from src.losses.mlm import MLMLoss
from src.model import Model
from src.simple_dataset import SimpleDataset
import torch.nn.functional as F
import torch

USE_CUDA = False

dataset = SimpleDataset()

device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

NUM_TOKENS = len(dataset.tokenizer.tokenizer)

model = Model(embed_size=80, num_hidden=80, num_layers=2, num_head=2, dropout=0.01, num_tokens=NUM_TOKENS)
model.to(device)
model.train()

mlm_loss = MLMLoss(model, pad_token_id=0, mask_token_id=1, mask_prob=0.5, random_token_prob=.1, num_tokens=NUM_TOKENS)

optimizer = torch.optim.Adagrad(model.parameters(), 0.01)

for epoch in range(500):
    print('Epoch', epoch + 1)

    for sentences, mask in dataset.iterator(entire_seq=True):
        for sent in sentences:
            print('Original:', ''.join(dataset.decode(sent)))
        inputs = torch.tensor(sentences).to(device)
        mask = torch.BoolTensor(mask).to(device)
        optimizer.zero_grad()

        loss = mlm_loss(inputs, mask)

        # Using this loss below works (look at src/model for insights)!
        # logits = model(inputs, mask)
        # loss = F.cross_entropy(
        #     logits.transpose(1, 2),
        #     inputs
        # )

        print(loss)
        loss.backward()
        optimizer.step()

        preds = model.decode(inputs, mask)
        preds = preds.cpu().detach().numpy()
        for pred, sent in zip(preds, sentences):
            print('Pred:', ''.join(dataset.decode(pred)), end='')
            if ''.join(dataset.decode(pred)) == ''.join(dataset.decode(sent)):
                print('   MATCHED!', end='')
            print('')

    print('')
