from src.config import Config
from src.datasets import BookDataset, DummyDataset
from src.logger import Logger
from src.pre_processing import TreeTokenizer, worker_init_fn
from src.utils import seed_torch
from src.model import AgentModel
from torch.utils.data.dataloader import DataLoader
import torch
import time

seed_torch(0)  # 0 learns 2 doesn't (before no cnn layer)

LOG_EVERY = 10
GENERATE_TEXT = False
PRINT_RECONSTRUCTED_TEXT = True


# Need to wrap in a function for the child workers
def train():
    dataset = DummyDataset(max_num=2)
    # dataset = BookDataset(no_stats=True, max_num=2)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        collate_fn=TreeTokenizer.batch_texts_to_trees,
        worker_init_fn=worker_init_fn,
        num_workers=0,
        # persistent_workers=True  # This is helpful when num_workers > 0
    )

    model = AgentModel()
    model.to(Config.device)

    main_params = [param for name, param in model.named_parameters() if
                   ("discriminator" not in name) and ("generator" not in name)]
    generator_params = [param for name, param in model.named_parameters() if "generator" in name]
    discriminator_params = [param for name, param in model.named_parameters() if "discriminator" in name]

    main_optimizer = torch.optim.Adam(main_params, 0.002)
    generator_optimizer = torch.optim.Adam(generator_params, 0.002)
    discriminator_optimizer = torch.optim.Adam(discriminator_params, 0.002)

    Logger.setup()
    for epoch in range(10001):
        print('Epoch', epoch + 1)
        # start_time = time.time()

        for batch in dataloader:
            model.train()
            main_optimizer.zero_grad()

            g_loss, disc_loss, main_loss, loss_object = model.forward(batch, generate=GENERATE_TEXT, epoch=epoch)
            Logger.log_losses(g_loss, disc_loss, main_loss, loss_object, step=epoch)
            Logger.log_l2_classifiers(model, step=epoch)

            if GENERATE_TEXT:
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
                model.eval()
                generated = {i: model.generate_texts(i, 1)[0] for i in reversed(range(Config.agent_level + 1))}

                nodes = batch.batch_root.children
                res = [model.full_decode(node) for node in nodes]
                reconstructed_text = [TreeTokenizer.deep_detokenize(r, Config.agent_level + 1) for r in res]
                sizes1 = [len(r) for r in res]  # should be [2,1]
                sizes2 = [[len(c.children) for c in r.children] for r in nodes]  # should be [2,1]
                sizes = {1: sizes1, 2: sizes2}

                if PRINT_RECONSTRUCTED_TEXT:
                    print(reconstructed_text)
                Logger.log_text(generated, reconstructed_text, sizes, step=epoch)
        # print('Epoch', epoch + 1, 'completed in', "%s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    train()
