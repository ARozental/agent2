from src.checkpoints import Checkpoints
from src.commands import Commands
from src.config import Config
from src.losses.calc import loss_object_to_main_loss, loss_object_to_reconstruction_weights_loss, \
    loss_object_to_extra_coherence_weights_loss
from src.datasets import BookDataset, DummyDataset, WikiDataset
from src.logger import Logger
from src.pre_processing import TreeTokenizer, worker_init_fn
from src.storage import Storage
from src.utils import seed_torch, merge_dicts, map_nested_dicts, metsumm
from src.model import AgentModel
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import time
import madgrad  # is it any good?
import torch.optim.lr_scheduler
import math

Commands.parse_arguments()

if Config.use_tpu:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

seed_torch(0)  # 0 learns 2 doesn't (before no cnn layer)

GENERATE_TEXT = False
PRINT_RECONSTRUCTED_TEXT = True

Config.setup_device()


# Need to wrap in a function for the child workers
def train(index, flags):
    if Config.use_tpu:
        Config.device = xm.xla_device()

    if Config.use_tpu and not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    if Config.use_dummy_dataset:
        dataset = DummyDataset(max_num=None)
    else:
        # dataset = BookDataset(no_stats=True, max_num=2)
        dataset = WikiDataset(max_num=None)

    if Config.use_tpu and xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

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
    coherence_params = [param for name, param in model.named_parameters() if "coherence" in name]
    reconstruction_params = [param for name, param in model.named_parameters() if
                             (("decompressor" in name) or ("decoder" in name))]

    if Config.optimizer == "Adam":
        main_optimizer = torch.optim.AdamW(main_params, Config.lr)
    else:
        main_optimizer = madgrad.MADGRAD(main_params, lr=Config.lr, momentum=Config.momentum)  # 0.01,0.9 is the default
    # main_optimizer = torch.optim.AdamW(main_params, 0.001) #todo: for dummy only

    lambda_lr = lambda batch: math.exp(math.log(0.5) / Config.half_life_steps) ** batch
    scheduler = torch.optim.lr_scheduler.LambdaLR(main_optimizer, lambda_lr)

    # generator_optimizer = torch.optim.AdamW(generator_params, 0.001)
    # discriminator_optimizer = torch.optim.AdamW(discriminator_params, 0.001)

    Storage.setup()
    Logger.setup()
    Checkpoints.setup()
    Checkpoints.load(model)
    all_times = []
    global_step = 0
    acc_loss_object = {0: {}, 1: {}}
    for epoch in range(10001):
        # print('Epoch', epoch + 1)
        
        if Config.use_tpu and Config.use_all_tpu_cores:
            parallel_loader = pl.ParallelLoader(dataloader, [Config.device]).per_device_loader(Config.device)
        else:
            parallel_loader = dataloader

        for step, batch in enumerate(parallel_loader):
            # This is not the most efficient, but needs to be done to not skip these examples in future epochs
            if Config.skip_batches is not None and (epoch == 0 and step < Config.skip_batches):
                global_step += 1
                continue

            if Config.use_tpu and Config.debug_tpu and (step == 0 or (step - 1) % Config.grad_acc_steps == 0):
                start_time = time.time()

            model.train()

            will_reconstruct = PRINT_RECONSTRUCTED_TEXT and (
                (epoch % (Config.grad_acc_steps * Config.log_every) == 0 and step == 0) or
                (step % (Config.grad_acc_steps * Config.log_every) == 0 and step > 0)
            )

            if Config.use_tpu:
                will_reconstruct = False  # Remove this once have the decode working on TPU

            g_loss, disc_loss, main_loss, loss_object = model.forward(batch, generate=GENERATE_TEXT,
                                                                      debug=will_reconstruct)
            acc_loss_object = merge_dicts(loss_object, acc_loss_object)

            main_loss = loss_object_to_main_loss(loss_object)
            r_loss = loss_object_to_reconstruction_weights_loss(loss_object)
            c_loss = loss_object_to_extra_coherence_weights_loss(loss_object)

            if GENERATE_TEXT:
                pass
                # todo: this is no longer up to date like the else part
            else:
                [setattr(p, "requires_grad", False) for p in main_params]
                [setattr(p, "requires_grad", True) for p in coherence_params]
                (c_loss / Config.grad_acc_steps).backward(retain_graph=True)
                [setattr(p, "requires_grad", False) for p in coherence_params]
                [setattr(p, "requires_grad", True) for p in reconstruction_params]
                (r_loss / Config.grad_acc_steps).backward(retain_graph=True)
                [setattr(p, "requires_grad", True) for p in main_params]
                (main_loss / Config.grad_acc_steps).backward()

                if Config.use_tpu:
                    xm.mark_step()

                # I want to clip on every step, how?
                if step % Config.grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(main_params, Config.grad_clip_value)
                    if Config.use_tpu and Config.use_all_tpu_cores:
                        xm.optimizer_step(main_optimizer)
                    else:
                        main_optimizer.step()
                    main_optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    if Config.use_tpu and not Config.use_all_tpu_cores:
                        xm.mark_step()

                    if step > 0:
                        acc_loss_object = map_nested_dicts(acc_loss_object, lambda x: x / Config.grad_acc_steps)

                    # Log
                    if Config.use_tpu:
                        xm.add_step_closure(Logger.log_losses,
                                            args=(g_loss, disc_loss, main_loss, acc_loss_object, global_step))
                        xm.add_step_closure(Logger.log_l2_classifiers, args=(model, global_step))
                    else:
                        Logger.log_losses(g_loss, disc_loss, main_loss, acc_loss_object, step=global_step)
                        Logger.log_l2_classifiers(model, step=global_step)

                    acc_loss_object = map_nested_dicts(acc_loss_object, lambda x: x * 0.0)

            # TODO - Take out the TPU blocker once printing reconstructed is working on TPU
            if not Config.use_tpu and (
                (epoch % (Config.grad_acc_steps * Config.log_every) == 0 and step == 0) or
                (step % (Config.grad_acc_steps * Config.log_every) == 0 and step > 0)
            ):
                print('Epoch', epoch, 'Batch', step)
                print(loss_object)
                print(main_loss)
                model.eval()

                if GENERATE_TEXT:
                    generated = {i: model.generate_texts(i, 1)[0] for i in reversed(range(Config.agent_level + 1))}
                    print(generated)
                    Logger.log_text(generated, step=global_step)

                if PRINT_RECONSTRUCTED_TEXT and not Config.use_tpu:  # TODO - Take out the TPU once working
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
                        if i == len(reconstructed) - 1:  # Upper most level
                            are_equal = [t == e for t, e in zip(text, expected)]
                            if False not in are_equal:
                                print('MATCHED')
                                exit()

                Checkpoints.save(model, epoch, global_step)

            if Config.use_tpu and Config.debug_tpu and step % Config.grad_acc_steps == 0:
                current_time = time.time() - start_time
                if global_step > 10:
                    all_times.append(current_time)
                    print('Step', global_step, 'completed in', round(current_time, 3), 'average',
                          round(np.mean(all_times), 3))
                else:
                    print('Step', global_step, 'completed in', round(current_time, 3))
                metsumm(global_step)
                print('')
                print('')

                # current_time = time.time() - start_time
        # all_times.append(current_time)
        # print('Epoch', epoch + 1, 'completed in', round(current_time, 3), 'average', round(np.mean(all_times), 3))


if __name__ == '__main__':
    if Config.use_tpu and Config.use_all_tpu_cores:
        flags = {}
        xmp.spawn(train, args=(flags,), nprocs=8, start_method='fork')
    else:
        train(None, None)
