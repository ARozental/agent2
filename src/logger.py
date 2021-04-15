from src.pre_processing import TreeTokenizer
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os


class Logger:
    writer = None
    viz = pd.DataFrame()
    loss_name_mapping = {
        'm': 'mlm',
        'c': 'coherence',
        'r': 'reconstruction',
        'e': 'eos',
        'j': 'join',
        'd': 'reconstruction_diff',
        'g': 'generator',
        'disc': 'discriminator',
    }

    @classmethod
    def setup(cls):
        cls.writer = SummaryWriter()

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

        for i, level in enumerate(model.agent_levels):
            cls.writer.add_scalar('l2/weight/' + str(i), level.classifier1w, step)
            cls.writer.add_scalar('l2/bias/' + str(i), level.classifier1b, step)

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
    def log_viz(cls, node, text, level, step):
        real_text = TreeTokenizer.deep_detokenize(node.build_struct(True)[0], node.level)
        Logger.viz = Logger.viz.append(pd.DataFrame({
            'text': [real_text],
            'pred': [text],
            'level': [level],
            'step': [step],
            'mlm': node.mlm_loss.item(),
        }), ignore_index=True)

        Logger.viz.to_csv(os.path.join('viz', 'results.csv'), index=False)
