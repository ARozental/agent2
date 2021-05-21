class FakeWrapper:
    def __init__(self, name, step_num=None):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyDebug:
    StepTrace = FakeWrapper
    Trace = FakeWrapper
