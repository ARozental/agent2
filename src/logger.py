from torch.utils.tensorboard import SummaryWriter


class Logger:
    writer = None

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
                cls.writer.add_scalar('losses/' + name + '/' + str(level), value, step)

    @classmethod
    def log_l2_classifiers(cls, model, step):
        for i, level in enumerate(model.agent_levels):
            cls.writer.add_scalar('l2/weight/' + str(i), level.classifier1w, step)
            cls.writer.add_scalar('l2/bias/' + str(i), level.classifier1b, step)

    @classmethod
    def log_text(cls, generated, reconstructed, sizes, step):
        for level, text in generated.items():
            cls.writer.add_text('generator/' + str(level), text, step)

        cls.writer.add_text('reconstructed', '  \n'.join(reconstructed), step)

        # for level, value in sizes.items():
        #     for i, val in enumerate(value):
        #         print(value, i, val)
        #         cls.writer.add_scalar('sizes/' + str(level) + '/' + str(i), val, step)
