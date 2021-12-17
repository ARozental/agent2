from src.debug.param_count import count_parameters
from src.checkpoints import Checkpoints
from src.commands import Commands
from src.config import Config
from src.debug.reconstruct import reconstruct_text
from src.losses.calc import loss_object_to_main_loss
from src.dataset import DummyDataset, WikiDataset, SimpleWikiDataset
from src.logger import Logger
from src.losses.rebalance import Rebalance
from src.pre_processing import worker_init_fn
from src.storage import Storage
from src.utils import seed_torch, metsumm, prepare_inputs, recycle_weights
from src.model import AgentModel
from src.debug.profiler import Profiler as xp
from torch.utils.data.dataloader import DataLoader, default_collate
import numpy as np
import time
import madgrad  # is it any good?
import torch.optim.lr_scheduler
import os
import collections
import torch
import torch.nn as nn

Commands.parse_arguments()
xp.setup()

if Config.use_tpu:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

    if Config.profile_tpu:
        os.environ['XLA_HLO_DEBUG'] = '1'

seed_torch(0)  # 0 learns 2 doesn't (before no cnn layer)

GENERATE_TEXT = False
PRINT_RECONSTRUCTED_TEXT = True

Config.setup_device()


def custom_collate_fn(data):
    return data[0][0], default_collate([data[0][1]])


# Need to wrap in a function for the child workers
def train(index, flags, training_started):
    if Config.use_tpu:
        Config.device = xm.xla_device()

    if Config.use_tpu and not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    if Config.use_dummy_dataset:
        dataset = DummyDataset(max_num=Config.max_dataset_len)
    elif Config.dataset == 'simple_wiki':
        dataset = SimpleWikiDataset(max_num=Config.max_dataset_len)
    else:
        # dataset = BookDataset(no_stats=True, max_num=2)
        dataset = WikiDataset(max_num=Config.max_dataset_len)

    if Config.use_tpu and xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_init_fn,
        num_workers=Config.num_dataset_workers,
        prefetch_factor=8,
        persistent_workers=True  # This is helpful when num_workers > 0
    )

    model = AgentModel()
    if Config.multi_gpu:
        model = nn.DataParallel(model, device_ids=[0, 1])

    if not Config.use_accelerator:
        model.to(Config.device)

    print([name for name, param in model.named_parameters() if ("discriminator" not in name) and ("generator" not in name)])
    main_params = [param for name, param in model.named_parameters() if
                   ("discriminator" not in name) and ("generator" not in name)]
    generator_params = [param for name, param in model.named_parameters() if "generator" in name]
    discriminator_params = [param for name, param in model.named_parameters() if "discriminator" in name]
    coherence_params = [param for name, param in model.named_parameters() if "coherence" in name]
    reconstruction_params = [param for name, param in model.named_parameters() if
                             (("decompressor" in name) or ("decoder" in name))]
    level0_tree_params = [param for name, param in model.named_parameters() if
                          ("discriminator" not in name) and
                          ("generator" not in name) and
                          ("decompressor" not in name) and
                          ("decoder" not in name) and
                          ("coherence_checker" not in name) and
                          ("encoder_transform" not in name) and
                          (("char_embedding_layer" in name) or ("agent_levels.0" in name))]

    if Config.optimizer == "Adam":
        if Config.use_8bit:
            import bitsandbytes as bnb
            main_optimizer = bnb.optim.Adam(main_params, Config.lr)
        else:
            main_optimizer = torch.optim.AdamW(main_params, Config.lr,amsgrad=True) #there are still some explosions with amsgrad but no drop before them....
    else:
        main_optimizer = madgrad.MADGRAD(main_params, lr=Config.lr, momentum=Config.momentum)  # 0.01,0.9 is the default
    # main_optimizer = torch.optim.AdamW(main_params, 0.001) #todo: for dummy only

    lambda_lr = lambda batch: Config.half_life_steps / (Config.half_life_steps + batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(main_optimizer, lambda_lr)

    # generator_optimizer = torch.optim.AdamW(generator_params, 0.001)
    # discriminator_optimizer = torch.optim.AdamW(discriminator_params, 0.001)

    if Config.use_accelerator:
        model, main_optimizer, dataloader = Config.accelerator.prepare(
            model, main_optimizer, dataloader
        )

    if Config.profile_tpu:
        server = xp.start_server(9012)

    Storage.setup()
    Logger.setup()
    Checkpoints.setup()
    Rebalance.setup()

    # print([name for name, param in model.named_parameters() if param.requires_grad])
    # for n,p in model.named_parameters():
    #   if "agent_levels.01" in n and p in main_params:
    #     p.requires_grad = False
    # print([name for name, param in model.named_parameters() if param.requires_grad])
    # 1+None
    #steal some old weights from trained models
    # with Storage.fs.open("models/trained_old.tar", 'rb') as f:
    #   trained_old = torch.load(f, map_location=torch.device('cpu'))
    # with Storage.fs.open("models/new_p_model.tar", 'rb') as f:
    #   untrained_new = torch.load(f,
    #                              map_location=torch.device('cpu'))
    # recycle_weights(untrained_new, trained_old)
    # torch.save(untrained_new, "recycled8.tar")
    # 1+None
    Checkpoints.load(model, main_optimizer, scheduler)
    all_times = []
    all_model_times = []
    global_step = 0
    print("freeze tree 0: ", Config.freeze0)
    if Config.freeze0:
        [setattr(p, "requires_grad", False) for p in level0_tree_params]
    if Config.skip_batches is not None:
        global_step = Config.skip_batches - 1
    count_parameters(model, trainable=True)
    for epoch in range(1000001):

        if Config.use_tpu and Config.use_all_tpu_cores:
            parallel_loader = pl.ParallelLoader(dataloader, [Config.device]).per_device_loader(Config.device)
        else:
            parallel_loader = dataloader

        total_loss = 0
        total_loss_object = {
            level: {'d': torch.tensor(0.0, device=Config.device)} for level in range(Config.agent_level + 1)
        }
        total_model_time = 0
        start_time = time.time()
        for step, (batch, inputs) in enumerate(parallel_loader):
            inputs = prepare_inputs(inputs, squeeze=True)

            grad_acc_steps = Config.grad_acc_fn(global_step)
            if Config.profile_tpu and step >= 4:
                training_started.set()

            # This is not the most efficient, but needs to be done to not skip these examples in future epochs
            # if Config.skip_batches is not None and (epoch == 0 and step < Config.skip_batches):
            #     global_step += 1
            #     continue

            current_model_time = time.time()
            model.train()
            will_reconstruct = PRINT_RECONSTRUCTED_TEXT and (
                (epoch % (grad_acc_steps * Config.log_every) == 0 and step == 0) or
                (step % (grad_acc_steps * Config.log_every) == 0 and step > 0)
            )
            if will_reconstruct:
              model.eval()

            if Config.use_tpu:
                will_reconstruct = False  # The TPU version computes the reconstruct vectors separately on the CPU

            # print(len(batch.level_nodes[0]),len(batch.level_nodes[1]),len(batch.level_nodes[0])/ len(batch.level_nodes[1]))# todo: this is for debug => fix it

            with xp.StepTrace('train_loop', step_num=step):
                noise_levels = torch.stack([total_loss_object[level]['d'] for level in range(Config.agent_level + 1)])
                g_loss, disc_loss, main_loss, loss_object, word_embedding_matrix, first_A1s, first_pndb_lookup_ids = model.forward(
                    batch,
                    inputs,
                    generate=GENERATE_TEXT,
                    debug=will_reconstruct,
                    noise_levels=noise_levels,
                    global_step=global_step,
                    xm=None if not Config.use_tpu else xm)
                if loss_object is None:
                    continue

                # TODO: Is this ok? Or do something else?
                if Config.multi_gpu:
                    g_loss = g_loss.mean()
                    disc_loss = disc_loss.mean()
                    main_loss = main_loss.mean()
                    for level in range(Config.agent_level + 1):
                        for key, value in loss_object[level].items():
                            loss_object[level][key] = value.mean()

                main_loss = loss_object_to_main_loss(loss_object) / grad_acc_steps
                # r_loss = loss_object_to_reconstruction_weights_loss(loss_object) / grad_acc_steps
                # c_loss = loss_object_to_extra_coherence_weights_loss(loss_object) / Config.grad_acc_steps

                # Divide by grad_acc_steps & detach from the graph
                loss_object = {
                    level_num: {key: value.detach() / grad_acc_steps for key, value in level.items()}
                    for level_num, level in loss_object.items()
                }

                # Sum up the loss objects
                total_loss_object = loss_object  # fix for when grad_acc > 1
                # if total_loss_object is None:
                #     total_loss_object = loss_object
                # else:
                #     total_loss_object = merge_dicts(total_loss_object, loss_object)

                # [setattr(p, "requires_grad", False) for p in main_params]
                # # [setattr(p, "requires_grad", True) for p in coherence_params]
                # # c_loss.backward(retain_graph=True)
                # # [setattr(p, "requires_grad", False) for p in coherence_params]
                # [setattr(p, "requires_grad", True) for p in reconstruction_params]
                # r_loss.backward(retain_graph=True)
                # [setattr(p, "requires_grad", True) for p in main_params]

                # main_loss.backward()

                total_loss += main_loss.detach()

                # if Config.use_tpu and not Config.profile_tpu:
                #    xm.mark_step()

                # TODO - I want to clip on every step, how?
                if (step + 1) % grad_acc_steps == 0:  # (step + 1) so that don't break on step 0 when acc is > 1
                    if Config.grad_clip_value and Config.grad_clip_value>0:
                      if Config.use_accelerator:
                          Config.accelerator.clip_grad_norm_(main_params, Config.grad_clip_value)
                      else:
                          torch.nn.utils.clip_grad_norm_(main_params, Config.grad_clip_value)

                    if Config.use_tpu:
                        xm.optimizer_step(main_optimizer)
                    else:
                        main_optimizer.step()
                    main_optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    # if Config.use_tpu and not Config.use_all_tpu_cores:
                    #     xm.mark_step()

                    # Log
                    if Config.use_tpu:
                        xm.add_step_closure(Logger.log_losses,
                                            args=(g_loss, disc_loss, total_loss, total_loss_object, global_step))
                        xm.add_step_closure(Logger.log_l2_classifiers, args=(model, global_step))
                    else:
                        Logger.log_losses(g_loss, disc_loss, main_loss, total_loss_object, step=global_step)
                        Logger.log_l2_classifiers(model, step=global_step)

                    # total_loss_object = None
                    # total_loss = 0

                Rebalance.add_loss_object(global_step, total_loss_object)
                if Config.rebalance_losses_step is not None and (global_step + 1) % Config.rebalance_losses_step == 0:
                    print('Rebalancing losses at step', global_step)  # This goes into effect on the next step
                    Rebalance.rebalance()
                    #main_optimizer.__setstate__({'state': collections.defaultdict(dict)})  # Clear the optimizer state

            total_model_time += (time.time() - current_model_time)

            if (not Config.use_accelerator or Config.accelerator.is_main_process) and \
                (epoch % (grad_acc_steps * Config.log_every) == 0 and step == 0) or \
                (step % (grad_acc_steps * Config.log_every) == 0 and step > 0):
                print('Epoch', epoch, 'Batch', step)
                if not Config.use_tpu:
                    # These are very inefficient for the TPU to print out
                    print(loss_object)
                    print(main_loss)
                model.eval()

                if Config.use_tpu:
                    # Moving model to CPU for evaluation
                    old_device = Config.device
                    Config.device = torch.device('cpu')
                    model.to(Config.device)
                    inputs = prepare_inputs(inputs, squeeze=False)
                    model.compute_vectors(batch, inputs)

                if GENERATE_TEXT:
                    generated = {i: model.generate_texts(i, 1)[0] for i in reversed(range(Config.agent_level + 1))}
                    print(generated)
                    Logger.log_text(generated, step=global_step)

                if PRINT_RECONSTRUCTED_TEXT:
                    reconstruct_text(batch, model, word_embedding_matrix, first_A1s, first_pndb_lookup_ids, global_step,
                                     exit_on_match=True)

                Checkpoints.save(epoch, global_step, model, main_optimizer, scheduler)

                if Config.use_tpu:
                    # Moving model back to TPU
                    Config.device = old_device
                    model.to(Config.device)

            if Config.use_tpu and Config.debug_tpu and step % grad_acc_steps == 0:
                current_time = time.time() - start_time
                current_model_time = total_model_time
                start_time = time.time()
                total_model_time = 0
                if global_step > 10:
                    all_times.append(current_time)
                    all_model_times.append(current_model_time)
                    print('Step', global_step, 'completed.')
                    print('Total time', round(current_time, 3), 'Average', round(np.mean(all_times), 3))
                    print('Model time', round(current_model_time, 3), 'Average', round(np.mean(all_model_times), 3))
                else:
                    print('Step', global_step, 'completed. Total time', round(current_time, 3), 'Model time',
                          round(current_model_time, 3))
                metsumm(global_step)
                print('')
                print('')

                # current_time = time.time() - start_time
        # all_times.append(current_time)
        # print('Epoch', epoch + 1, 'completed in', round(current_time, 3), 'average', round(np.mean(all_times), 3))


def target_fn(training_started):
    # sys.stdout = open('training_logs.stdout', 'w')
    # sys.stderr = open('training_logs.stderr', 'w')
    xmp.spawn(train, args=({}, training_started,), nprocs=8, start_method='fork')


def target_single_fn(training_started):
    train(None, None, training_started)


if __name__ == '__main__':
    if Config.use_tpu and Config.use_all_tpu_cores:
        if Config.profile_tpu:
            if not Config.log_experiment:
                raise ValueError('"log_experiment" needs to be turned on for the profiler to write to Tensorboard.')

            import multiprocessing

            training_started = multiprocessing.Event()
            p = multiprocessing.Process(target=target_fn, args=(training_started,))
            p.start()

            training_started.wait(120 * 10)

            import re

            tpu_ip = re.match('grpc\://((\d{1,3}\.){3}\d{1,3})\:\d{4}', os.environ.get('TPU_NAME')).group(1)
            xp.trace('localhost:9012', Logger.get_log_dir())  # client side profiling
            xp.trace(f'{tpu_ip}:8466', Logger.get_log_dir())  # need GCS bucket for all traces to be written
        else:
            flags = {}
            xmp.spawn(train, args=(flags, None,), nprocs=8, start_method='fork')
    else:
        if Config.profile_tpu:
            if not Config.log_experiment:
                raise ValueError('"log_experiment" needs to be turned on for the debugger to write to Tensorboard.')

            import multiprocessing

            training_started = multiprocessing.Event()
            p = multiprocessing.Process(target=target_single_fn, args=(training_started,))
            p.start()

            training_started.wait(120 * 10)

            import re

            tpu_ip = re.match('grpc\://((\d{1,3}\.){3}\d{1,3})\:\d{4}', os.environ.get('TPU_NAME')).group(1)
            xp.trace('localhost:9012', Logger.get_log_dir(), duration_ms=11 * 1000)  # client side profiling
            xp.trace(f'{tpu_ip}:8466', Logger.get_log_dir(),
                     duration_ms=11 * 1000)  # need GCS bucket for all traces to be written
        else:
            train(None, None, None)
