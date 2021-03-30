from src.config import Config
from src.datasets import BookDataset, DummyDataset
from src.logger import Logger
from src.utils import seed_torch
from src.model import AgentModel
from src.tree_dataset import TreeDataset
from src.pre_processing import TreeTokenizer
import torch

seed_torch(0)  # 0 learns 2 doesn't (before no cnn layer)

LOG_EVERY = 2
PRINT_RECONSTRUCTED_TEXT = True

dataset = DummyDataset(batch_size=2)
# dataset = BookDataset(batch_size=2, no_stats=True)

model = AgentModel(TreeTokenizer())
model.to(Config.device)
# model.train()

main_params = [param for name, param in model.named_parameters() if
               ("discriminator" not in name) and ("generator" not in name)]
generator_params = [param for name, param in model.named_parameters() if "generator" in name]
discriminator_params = [param for name, param in model.named_parameters() if "discriminator" in name]

main_optimizer = torch.optim.Adam(main_params, 0.002)
generator_optimizer = torch.optim.Adam(generator_params, 0.002)
discriminator_optimizer = torch.optim.Adam(discriminator_params, 0.002)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
# optimizer_D = torch.optim.AdamW(D.parameters(), 0.001)
# d_loss.backward(retain_graph=True)
# print("discriminator_param: ", discriminator_params[0].data[0][0])

Logger.setup()
for epoch in range(10001):
    print('Epoch', epoch + 1)
    generate = True

    for batch in dataset:
        model.train()
        main_optimizer.zero_grad()

        g_loss, disc_loss, main_loss, loss_object = model.forward(batch, generate=generate, epoch=epoch)
        Logger.log_losses(g_loss, disc_loss, main_loss, loss_object, step=epoch)
        Logger.log_l2_classifiers(model, step=epoch)

        if generate:
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            main_loss.backward(retain_graph=True)
            [setattr(p, "requires_grad", False) for p in main_params + generator_params]
            disc_loss.backward(retain_graph=True)
            [setattr(p, "requires_grad", True) for p in generator_params]
            [setattr(p, "requires_grad", False) for p in discriminator_params]
            (g_loss - disc_loss * 0.2).backward()  # disc loss won't go down even when this is commented => BUG
            [setattr(p, "requires_grad", True) for p in main_params + discriminator_params]
            main_optimizer.step()
            discriminator_optimizer.step()
            generator_optimizer.step()
        else:
            main_loss.backward()
            main_optimizer.step()

        if epoch % LOG_EVERY == 0:
            generated = {i: model.generate_texts(i, 1)[0] for i in reversed(range(3))}

            nodes = batch.batch_root.children
            res = [model.full_decode(node) for node in nodes]
            reconstructed_text = [model.tree_tokenizer.deep_detokenize(r, 3) for r in res]
            sizes1 = [len(r) for r in res]  # should be [2,1]
            sizes2 = [[len(c.children) for c in r.children] for r in nodes]  # should be [2,1]
            sizes = {1: sizes1, 2: sizes2}

            if PRINT_RECONSTRUCTED_TEXT:
                print(reconstructed_text)
            Logger.log_text(generated, reconstructed_text, sizes, step=epoch)
