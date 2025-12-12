from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton


# ============================= CONFIGURATION COMPONENTS =============================
@dataclass(frozen=True)
class KernelsConfigV2:
    """Configuration container for FlashAttention-2 Triton kernels.

    Encapsulate all parameters needed for compiling and executing
    forward and backward attention kernels with Triton.

    Attributes:
        block_rows_size: Block size for query sequence dimension (forward only).
        block_cols_size: Block size for key/value sequence dimension (forward only).
        min_block_headdim: Minimum block size for head dimension (must be power of 2,
            at least 16).
        max_headdim: Maximum supported head dimension (kernel constraint).
        seqlen_cache_divisor: Sequence length quantizer (limit number of compilations
            (most common: 32)).
        min_warps: Minimum number of warps for kernel execution (GPU-specific).
        max_warps: Maximum number of warps for kernel execution (GPU-specific).
        num_stages: Number of pipelining stages for kernel execution (GPU-specific).
        backward_autotune_configs: Triton autotune configurations for backward pass.
    """

    block_rows_size: int
    block_cols_size: int
    min_block_headdim: int
    max_headdim: int
    seqlen_cache_divisor: int
    min_warps: int
    max_warps: int
    num_stages: int
    backward_autotune_configs: list[triton.Config]

    def _create_kernel_key(self) -> tuple[tuple[int, int], tuple]:  # type: ignore[type-arg]
        """Create an immutable cache key needed for the backward kernel configuration.

        Returns:
            Tuple containing:
            - GPU compute capability (major, minor) for a current device.
            - Tuple of all configuration values.
        """
        cc = torch.cuda.get_device_capability()

        # Internal helper for field processing based on its name
        def _serialize(name: str) -> object:
            attr = getattr(self, name)
            return self._serialize_bwd(attr) if name == "backward_autotune_configs" else attr

        return cc, tuple(_serialize(name) for name in self.__annotations__)

    def _serialize_bwd(self, bwd_configs: list[triton.Config]) -> tuple:  # type: ignore[type-arg]
        """Serialize Triton autotune configurations to a hashable format.

        Args:
            bwd_configs: List of Triton autotune backward configurations.

        Returns:
            Tuple of serialized configurations, where each configuration contains:
            - Tuple of kwargs values.
            - Number of warps.
            - Number of stages.
            - Pre-hook function qualname (if callable).
        """
        return tuple(
            (tuple(c.kwargs.values()), c.num_warps, c.num_stages, self._hook_name(c.pre_hook))
            for c in bwd_configs
        )

    @staticmethod
    def _hook_name(pre_hook: object) -> str | None:
        """Extract a unique identifier for a hook function if available, None otherwise."""
        return getattr(pre_hook, "__qualname__", None)


def init_to_zero_v2(name: str) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Pre-hook for triton.Config.

    Used in backward autotuning that initializes a tensor in nargs to zero by name.

    Args:
        name: Key identifying the tensor to be zero-initialized in the kernel arguments
                dictionary (e.g., "DQ" for query gradient, "DK" for key gradient).

    Returns:
        A function that zeros out the specified tensor inplace and then returns it.
    """
    return lambda nargs: nargs[name].zero_()
