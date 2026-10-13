from ..base import MetalBackend
from .attention import finalize, validate


class Backend(MetalBackend):
    """Measured 12-core Apple11 configuration, independent of M5 fusion recipes."""
    id = "m6"
    finalize = staticmethod(finalize)
    validate_config = staticmethod(validate)

    def serving_context_limit(self, *, drafter) -> int | None:
        # On the tested 32 GB M6, 32K with 128-row prefill fails GPU residency.
        # 16K generates successfully, but saving its host prefix checkpoint
        # coincides with swapping. Retain 8K with the default prefix-cache policy.
        return 8192 if drafter else None

    def serving_prefill_chunk(self, recipes, model=None):
        # With validated matrix acceleration, 128-row prompt passes complete
        # the long-prompt warmup and outperform 8/32-row chunks on this device.
        return 128
