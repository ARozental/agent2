from src.config import Config
from fsspec.implementations.local import LocalFileSystem
import gcsfs


class Storage:
    fs = None

    @classmethod
    def setup(cls):
        if Config.storage_location is not None:
            cls.fs = gcsfs.GCSFileSystem()
        else:
            cls.fs = LocalFileSystem()
