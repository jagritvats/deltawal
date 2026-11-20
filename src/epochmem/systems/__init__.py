from .b1_sync import B1Sync
from .b2_async import B2Async
from .b3_recency import B3Recency
from .b4_epoch_wal import B4EpochWAL

REGISTRY = {cls.name: cls for cls in (B1Sync, B2Async, B3Recency, B4EpochWAL)}
