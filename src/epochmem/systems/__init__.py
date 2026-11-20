from .b1_sync import B1Sync, B1SyncVersioned
from .b2_async import B2Async
from .b3_recency import B3Recency, B3RecencyMatched, B3RecencyMatchedUser
from .deltawal import DeltaWALLazy, DeltaWALPrompt, DeltaWALShadow, DeltaWALShadowV2

REGISTRY = {cls.name: cls for cls in (B1Sync, B1SyncVersioned, B2Async, B3Recency, B3RecencyMatched, B3RecencyMatchedUser,
                                      DeltaWALPrompt, DeltaWALShadow,
                                      DeltaWALShadowV2, DeltaWALLazy)}
