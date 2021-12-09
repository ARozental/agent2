from src.pre_processing import TreeTokenizer
from src.config import Config
from src.storage import Storage

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os


class Logger:
    writer = None
    viz = pd.DataFrame()
    loss_name_mapping = {
        'm': 'mlm',
        #'md': 'mlm_diff',
        'c': 'coherence',
        'r': 'reconstruction',
        'e': 'eos',
        'j': 'join',
        'd': 'reconstruction_diff',
        'rc': 'reconstruction_coherence',
        're': 'reconstruction_eos',
        'rj': 'reconstruction_join',
        'rm': 'reconstruction_mlm',
        #'rmd': 'reconstruction_diff_mlm',
        'g': 'generator',
        'disc': 'discriminator'
        #'cd': 'coherence_discriminator',
        #'rcd': 'reconstruction_coherence_discriminator'

    }

    @classmethod
    def setup(cls):
        if Config.log_experiment:
            cls.writer = SummaryWriter(log_dir=cls.get_log_dir())

    @staticmethod
    def get_log_dir():
        log_dir = []
        if Config.storage_location is not None:
            log_dir.append(Config.storage_location)
            if Config.exp_folder is None:
                # Taken from PyTorch SummaryWriter source
                import socket
                from datetime import datetime
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                log_dir.append(os.path.join('runs', current_time + '_' + socket.gethostname()))

        if Config.exp_folder is not None:
            log_dir.append(Config.exp_folder)

        if len(log_dir) == 0:
            log_dir = None
        else:
            log_dir = os.path.join(*log_dir)

        return log_dir

    @classmethod
    def log_losses(cls, g_loss, disc_loss, main_loss, loss_object, step):
        if cls.writer is None:
            return

        cls.writer.add_scalar('loss/generator', g_loss, step)
        cls.writer.add_scalar('loss/discriminator', disc_loss, step)
        cls.writer.add_scalar('loss/main', main_loss, step)

        for level, losses in loss_object.items():
            for name, value in losses.items():
                name = cls.loss_name_mapping[name]
                cls.writer.add_scalar('losses/' + name + '/' + str(level), value, step)

    @classmethod
    def log_l2_classifiers(cls, model, step):
        if cls.writer is None:
            return

        if Config.use_accelerator:
            Config.accelerator.wait_for_everyone()
            model = Config.accelerator.unwrap_model(model)

        if Config.multi_gpu:
            model = model.module

        for i, level in enumerate(model.agent_levels):
            cls.writer.add_scalar('l2/weight/' + str(i), level.classifier1w, step)
            cls.writer.add_scalar('l2/bias/' + str(i), level.classifier1b, step)

            cls.writer.add_scalar('l2/join/weight/' + str(i), level.join_classifier_w, step)
            cls.writer.add_scalar('l2/join/bias/' + str(i), level.join_classifier_b, step)

    @classmethod
    def log_text(cls, generated, step):
        if cls.writer is None:
            return

        for level, text in generated.items():
            cls.writer.add_text('generator/' + str(level), text, step)

    @classmethod
    def log_reconstructed(cls, text, level, step):
        if cls.writer is None:
            return

        cls.writer.add_text('reconstructed/' + str(level), '  \n'.join(text), step)

    @classmethod
    def log_reconstructed_e(cls, text, level, step):
        if cls.writer is None:
            return

        cls.writer.add_text('reconstructed_e/' + str(level), '  \n'.join(text), step)

    @classmethod
    def log_pndb(cls, averages, step):
        if cls.writer is None:
            return

        cls.writer.add_scalar('pndb/avg_all_act', averages['all'], step)
        cls.writer.add_scalar('pndb/avg_proper_act', averages['proper'], step)
        cls.writer.add_scalar('pndb/avg_non_act', averages['non'], step)

    @classmethod
    def log_viz(cls, node, text, level, step):
        if Config.viz_file is None:
            return

        real_text = TreeTokenizer.deep_detokenize(node.build_struct(True)[0], node.level)
        Logger.viz = Logger.viz.append(pd.DataFrame({
            'text': [real_text],
            'pred': [text],
            'level': [level],
            'step': [step],
            'mlm': node.mlm_loss.item(),
            'mlm_diff': node.mlm_diff_loss.item(),
            'coherence': node.coherence_loss.item(),
            'eos': node.eos_loss.item(),
            'join': node.join_loss.item(),
            'recon': node.reconstruction_loss.item(),
            'recon_diff': node.reconstruction_diff_loss.item(),
            'rc': node.rc_loss.item(),
            're': node.re_loss.item(),
            'rj': node.rj_loss.item(),
            'rm': node.rm_loss.item(),
            'rm_diff': node.rm_diff_loss.item(),
        }), ignore_index=True)

        with Storage.fs.open(Config.viz_file, 'w') as f:
            Logger.viz.to_csv(f, index=False)
