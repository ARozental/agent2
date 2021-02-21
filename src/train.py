from src.losses.mlm import MLMLoss
from src.model import Model
from src.simple_dataset import SimpleDataset
import torch.nn.functional as F
import torch

dataset = SimpleDataset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_TOKENS = len(dataset.tokenizer.tokenizer)

model = Model(embed_size=40, num_hidden=10, num_layers=1, num_head=1, dropout=0.2, num_tokens=NUM_TOKENS)
model.to(device)
model.train()

mlm_loss = MLMLoss(model, pad_token_id=0, mask_token_id=1, mask_prob=0.1, random_token_prob=.1, num_tokens=NUM_TOKENS)

optimizer = torch.optim.Adagrad(model.parameters())

for epoch in range(50):
    print('Epoch', epoch + 1)

    for word in dataset.iterator(entire_seq=True):
        word_text = ''.join(dataset.decode(word))
        print(word_text)
        word = torch.tensor([word]).to(device)
        mask = torch.ones((1, word.size(0))).to(device)  # Ignoring mask inside model for now
        optimizer.zero_grad()

        # This doesn't work!
        # loss = mlm_loss(word, mask)

        # Using this loss below works (look at src/model for insights)!
        logits = model(word, mask)
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            word
        )

        print(loss)
        loss.backward()
        optimizer.step()

        pred = model.decode(word, mask)
        pred = pred.cpu().detach().numpy()
        pred = pred[0]
        print(''.join(dataset.decode(pred)))
        if ''.join(dataset.decode(pred)) == word_text:
            print('MATCHED!')
            # exit()
    print('')
