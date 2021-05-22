from src.config import Config


class FakeWrapper:
    def __init__(self, name, step_num=None):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Profiler:
    StepTrace = None
    Trace = None
    trace = None
    start_server = None

    @classmethod
    def setup(cls):
        if Config.profile_tpu:
            import torch_xla.debug.profiler as xp
            cls.StepTrace = xp.StepTrace
            cls.Trace = xp.Trace
            cls.trace = xp.trace
            cls.start_server = xp.start_server
        else:
            cls.StepTrace = FakeWrapper
            cls.Trace = FakeWrapper
