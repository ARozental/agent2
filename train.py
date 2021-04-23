from src.config import Config
from src.datasets import BookDataset, DummyDataset, WikiDataset
from src.logger import Logger
from src.pre_processing import TreeTokenizer, worker_init_fn
from src.utils import seed_torch
from src.model import AgentModel
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import time
import os

seed_torch(0)  # 0 learns 2 doesn't (before no cnn layer)

MODEL_FOLDER = os.path.join('models', Config.model_folder)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
GENERATE_TEXT = False
PRINT_RECONSTRUCTED_TEXT = True


# Need to wrap in a function for the child workers
def train():
    # dataset = DummyDataset(max_num=None)
    # dataset = BookDataset(no_stats=True, max_num=2)
    dataset = WikiDataset(max_num=None)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        collate_fn=TreeTokenizer.batch_texts_to_trees,
        worker_init_fn=worker_init_fn,
        num_workers=1,
        persistent_workers=True  # This is helpful when num_workers > 0
    )

    model = AgentModel()
    model.to(Config.device)

    main_params = [param for name, param in model.named_parameters() if
                   ("discriminator" not in name) and ("generator" not in name)]
    generator_params = [param for name, param in model.named_parameters() if "generator" in name]
    discriminator_params = [param for name, param in model.named_parameters() if "discriminator" in name]

    main_optimizer = torch.optim.AdamW(main_params, 0.0005)
    generator_optimizer = torch.optim.AdamW(generator_params, 0.001)
    discriminator_optimizer = torch.optim.AdamW(discriminator_params, 0.001)

    Logger.setup()
    all_times = []
    global_step = 0
    for epoch in range(10001):
        # print('Epoch', epoch + 1)
        # start_time = time.time()

        for batch_num, batch in enumerate(dataloader):
            model.train()
            main_optimizer.zero_grad()

            will_reconstruct = PRINT_RECONSTRUCTED_TEXT and (
                    (epoch % Config.log_every == 0 and batch_num == 0) or
                    (batch_num % Config.log_every == 0 and batch_num > 0)
            )

            g_loss, disc_loss, main_loss, loss_object = model.forward(batch, generate=GENERATE_TEXT,
                                                                      debug=will_reconstruct)
            Logger.log_losses(g_loss, disc_loss, main_loss, loss_object, step=global_step)
            Logger.log_l2_classifiers(model, step=global_step)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip_value)
                main_optimizer.step()
                discriminator_optimizer.step()
                generator_optimizer.step()
            else:
                torch.nn.utils.clip_grad_value_(main_params, Config.grad_clip_value)
                main_loss.backward()
                main_optimizer.step()

            if (epoch % Config.log_every == 0 and batch_num == 0) or \
                    (batch_num % Config.log_every == 0 and batch_num > 0):
                print('Epoch', epoch, 'Batch', batch_num)
                #print(loss_object)
                model.eval()

                if GENERATE_TEXT:
                    generated = {i: model.generate_texts(i, 1)[0] for i in reversed(range(Config.agent_level + 1))}
                    print(generated)
                    Logger.log_text(generated, step=global_step)

                if PRINT_RECONSTRUCTED_TEXT:
                    nodes = batch.batch_root.children
                    expected = [TreeTokenizer.deep_detokenize(node.build_struct(return_eos=True)[0], Config.agent_level)
                                for node in nodes]
                    reconstructed = [model.full_decode(batch.level_nodes[i][:5]) for i in range(Config.agent_level + 1)]
                    reconstructed = [[TreeTokenizer.deep_detokenize(node[0], i) for node in items] for i, items in
                                     enumerate(reconstructed)]
                    for i, text in enumerate(reconstructed):
                        print('Level', i, text)
                        Logger.log_reconstructed(text, i, step=global_step)
                        for j, item in enumerate(text):
                            Logger.log_viz(batch.level_nodes[i][j], text[j], i, step=global_step)
                        if i == len(reconstructed) - 1:
                            if text[0] == expected[0] and text[1] == expected[1]:
                                print('MATCHED')
                                exit()

            if Config.save_every is not None and batch_num > 0 and batch_num % Config.save_every == 0:
                torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, str(epoch) + '.' + str(batch_num)))

            global_step += 1

        # current_time = time.time() - start_time
        # all_times.append(current_time)
        # print('Epoch', epoch + 1, 'completed in', round(current_time, 3), 'average', round(np.mean(all_times), 3))


if __name__ == '__main__':
    train()
