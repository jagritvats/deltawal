from .b1_sync import B1Sync
from .b2_async import B2Async
from .b3_recency import B3Recency
from .deltawal import DeltaWALLazy, DeltaWALPrompt, DeltaWALShadow, DeltaWALShadowV2

REGISTRY = {cls.name: cls for cls in (B1Sync, B2Async, B3Recency, DeltaWALPrompt, DeltaWALShadow,
                                      DeltaWALShadowV2, DeltaWALLazy)}
