from src.config import MODEL_CONFIG
from src.losses import MLMLoss, CoherenceLoss, ReconstructionLoss
from src.model import Model
from src.simple_dataset import SimpleDataset
import torch

USE_CUDA = False

dataset = SimpleDataset()

device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

NUM_TOKENS = len(dataset.tokenizer.tokenizer)

model = Model(MODEL_CONFIG, num_tokens=NUM_TOKENS, max_seq_length=dataset.max_length)
model.to(device)
model.train()

losses = {
    'mlm': MLMLoss(model.levels[0],
                   pad_token_id=dataset.tokenizer.SPECIAL_INDICES['[PAD]'],
                   mask_token_id=dataset.tokenizer.SPECIAL_INDICES['[MASK]'],
                   mask_prob=0.5,
                   random_token_prob=0.1,
                   num_tokens=NUM_TOKENS),
    'coherence': CoherenceLoss(model.levels[0]),
    'reconstruct': ReconstructionLoss(model.levels[0]),
}

optimizer = torch.optim.Adagrad(model.parameters(), 0.01)

for epoch in range(500):
    print('Epoch', epoch + 1)

    for sentences, mask in dataset.iterator():
        for sent in sentences:
            print('Original:', ''.join(dataset.decode(sent)))
        inputs = torch.tensor(sentences).to(device)
        mask = torch.BoolTensor(mask).to(device)
        optimizer.zero_grad()

        mlm_loss = losses['mlm'](inputs, mask)
        coherence_loss = losses['coherence'](inputs, mask)
        reconstruct_loss = losses['reconstruct'](inputs, mask)

        print('mlm_loss', mlm_loss.item())
        print('coherence_loss', coherence_loss.item())
        print('reconstruct_loss', reconstruct_loss.item())
        (mlm_loss + coherence_loss + reconstruct_loss).backward()
        optimizer.step()

        with torch.no_grad():
            preds = model.levels[0].decode(inputs, mask)
        preds = preds.cpu().detach().numpy()
        for pred, sent in zip(preds, sentences):
            print('Pred:', ''.join(dataset.decode(pred)), end='')
            if ''.join(dataset.decode(pred)) == ''.join(dataset.decode(sent)):
                print('   MATCHED!', end='')
            print('')

    print('')
