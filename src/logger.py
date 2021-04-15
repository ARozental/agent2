from src.config import Config

from torch.utils.tensorboard import SummaryWriter


class Logger:
    writer = None
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
        if Config.log_experiment:
            cls.writer = SummaryWriter(log_dir=Config.exp_folder)

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
