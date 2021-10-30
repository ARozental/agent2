from src.config import Config


class Rebalance:
    losses = None

    @classmethod
    def setup(cls):
        if Config.rebalance_losses_step is None:
            return

        if Config.rebalance_losses_aggregate > Config.rebalance_losses_step:
            raise ValueError('Config.rebalance_losses_aggregate must be less than Config.rebalance_losses_step')

    @classmethod
    def add_loss_object(cls, step, loss_object):
        if Config.rebalance_losses_step is None:
            return

        if 0 != (step + 1) % Config.rebalance_losses_step < Config.rebalance_losses_aggregate:
            return

        if cls.losses is None:
            cls.losses = loss_object
        else:
            cls.losses = {i: {k: v1 + v2 for (k, v1), (_, v2) in zip(cls.losses[i].items(), loss_object[i].items())}
                          for i in range(Config.agent_level + 1)}

    @classmethod
    def rebalance(cls):
        for i in range(Config.agent_level + 1):
            level_weights = cls.get_level_weights(i)
            for name, value in level_weights.items():
                Config.loss_weights[i][name] = value / cls.losses[i][name]

        cls.losses = None  # Reset the aggregation

    @classmethod
    def get_level_weights(cls, i):
        loss_names = set(cls.losses[i].keys()) #maybe make sure none of them are 0 here as well?
        num_empty_weights = len(loss_names - set(Config.rebalance_percentages[i].keys()))
        total_weights = [val for key, val in Config.rebalance_percentages[i].items() if val > 0 and key in loss_names]
        leftover_weight = 1.0 - sum(total_weights)
        weight_per = leftover_weight / num_empty_weights

        return {name: Config.rebalance_percentages[i].get(name, weight_per) for name in loss_names}
