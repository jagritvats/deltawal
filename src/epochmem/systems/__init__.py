from .b1_sync import B1Sync, B1SyncVersioned
from .b2_async import B2Async
from .b3_recency import B3Recency, B3RecencyMatched, B3RecencyMatchedUser
from .deltawal import DeltaWALLazy, DeltaWALPrompt, DeltaWALShadow, DeltaWALShadowV2, DeltaWALVersioned

REGISTRY = {cls.name: cls for cls in (B1Sync, B1SyncVersioned, B2Async, B3Recency, B3RecencyMatched, B3RecencyMatchedUser,
                                      DeltaWALPrompt, DeltaWALShadow,
                                      DeltaWALShadowV2, DeltaWALLazy, DeltaWALVersioned)}

# Additive: DeltaWAL-Versioned ablations and prompt probes (systems/ablations.py); existing entries unchanged.
from .ablations import ABLATION_SYSTEMS  # noqa: E402

REGISTRY.update({cls.name: cls for cls in ABLATION_SYSTEMS})
