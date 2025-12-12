"""FlashAttention-2 Triton implementation.

Based on Tri Dao's paper "FlashAttention-2:Faster Attention with Better Parallelism
and Work Partitioning".

Key Features:
- Cross-platform support (Linux and Windows).
- Dual-mode operation: deterministic (sequence-parallel disabled) and
    non-deterministic (higher performance).
- Hardware-aware optimizations for Turing (CC 7.5) and Ampere+ (CC 8.0+) architectures.
- Custom configuration support for older GPU architectures or specialized tuning.
- Support for homo and heterogeneous GPU clusters with automatic configuration selection.

Compatibility Notice:
- Blackwell GPUs require PyTorch 2.7.0+, CUDA 12.8+, Triton 3.3.0+.
- For non-Blackwell GPUs (Turing-Hopper only) is recommened to use legacy mode for
    peak performance via:
        pip install flash_attention_triton[legacy].
- Triton versions 3.3.0+ (at the moment 3.3.0-3.5.0) have issues (bugs) with
    increased shared memory usage on pre-Blackwell architectures (notably for Turing,
    reducing its performance to vanilla attention).
- Tested with required dependencies by installation mode:
    Legacy (Turing-Hopper): PyTorch 2.5.0-2.6.0, CUDA 11.8+, Triton 3.1.0-3.2.0.
    Modern (Turing-Blackwell): PyTorch 2.7.0+, CUDA 12.8+, Triton 3.3.0+.

Caution:
- Non-deterministic mode may yield slightly different results across runs due to
    sequence-parallel (atomic) operations.
- For production deployments requiring reproducibility, use the deterministic mode flag.
- Performance characteristics may vary across GPU architectures.
"""

import math
import warnings

import torch
import triton
from packaging import version

from ._configurators_v2 import KernelsConfigV2, init_to_zero_v2
from ._triton_kernels_v2 import (
    _backward_kernel,
    _backward_preprocess_do_o_dot,
    _forward_kernel,
)

# Min new and max old allowed triton versions for different scenarios
_OLD_TRITON_VERSION = version.parse("3.2.0")
_NEW_TRITON_VERSION = version.parse("3.3.0")
_CURRENT_TRITON_VERSION = version.parse(triton.__version__)

# Compute capabilities for different GPU architectures
_TURING_CC = (7, 5)
_BLACKWELL_CC = (9, 0)

# Flag indicating if the Turing GPU Triton compatibility warning has been displayed
_TURING_WARN_SHOWN = False


# ================================ AUTOGRAD INTEGRATION ================================
def _flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    block_rows_size: int,
    block_cols_size: int,
    min_block_headdim: int,
    max_headdim: int,
    seqlen_cache_divisor: int,
    min_warps: int,
    max_warps: int,
    num_stages: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Internal implementation of FlashAttention-2 forward pass.

    Validate inputs, allocates buffers, and launche optimized Triton kernel.
    """
    # Shape constraints
    batch_idx, nheads_idx, seqlen_idx = 0, 1, 2
    batch, nheads, seqlen_q, headdim = q.shape
    _, _, seqlen_k, _ = k.shape

    # Validate tensor shapes and properties
    assert k.shape == (batch, nheads, seqlen_k, headdim)
    assert v.shape == (batch, nheads, seqlen_k, headdim)
    assert headdim <= max_headdim, f"Supports head dimensions only up to {max_headdim}"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All tensors must reside on CUDA device"
    assert q.dtype == k.dtype == v.dtype, "All tensors must be the same type"
    assert q.dtype == torch.float16, "Supports only fp16"

    # Prepare output tensor and softmax scale
    output = torch.empty_like(q)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)

    # Allocate log-sum-exp (lse) and scratchpad (spb) buffers using the query sequence length
    # aligned to the nearest greater multiple of max_headdim
    seqlen_q_rounded = math.ceil(seqlen_q / max_headdim) * max_headdim
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    spd = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    # Configure kernel launch parameters
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_Q_ROWS_SIZE"]), batch * nheads)
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), min_block_headdim)
    num_warps = min_warps if headdim <= max_headdim else max_warps

    # Launch Triton kernel with defined grid for parallel block-based computations:
    # - Dim0: Blocks along query sequence
    # - Dim1: Batch-head combinations
    _forward_kernel[grid](
        q,
        k,
        v,
        output,
        lse,
        spd,
        softmax_scale,
        q.stride(batch_idx),
        q.stride(nheads_idx),
        q.stride(seqlen_idx),
        k.stride(batch_idx),
        k.stride(nheads_idx),
        k.stride(seqlen_idx),
        v.stride(batch_idx),
        v.stride(nheads_idx),
        v.stride(seqlen_idx),
        output.stride(batch_idx),
        output.stride(nheads_idx),
        output.stride(seqlen_idx),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        seqlen_q // seqlen_cache_divisor,
        seqlen_k // seqlen_cache_divisor,
        BLOCK_HEADDIM,
        BLOCK_Q_ROWS_SIZE=block_rows_size,
        BLOCK_KV_COLS_SIZE=block_cols_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    d_output: torch.Tensor,
    softmax_scale: float | None,
    kernels_config: KernelsConfigV2,
) -> None:
    """Internal implementation of FlashAttention-2 backward pass.

    Coordinate gradient computation through 2 specialized Triton kernels.
    """
    # Shape constraints
    batch_idx, nheads_idx, seqlen_idx = 0, 1, 2
    batch, nheads, seqlen_q, headdim = q.shape
    _, _, seqlen_k, _ = k.shape

    # Make sure if the last dimension is contiguous
    if not d_output.is_contiguous():
        d_output = d_output.contiguous()

    # Extract the necessary parameters from the config
    block_rows_size = kernels_config.block_rows_size
    min_block_headdim = kernels_config.min_block_headdim
    max_headdim = kernels_config.max_headdim
    seqlen_cache_divisor = kernels_config.seqlen_cache_divisor

    # Align the query sequence length to the nearest greater multiple of max_headdim
    seqlen_q_rounded = math.ceil(seqlen_q / max_headdim) * max_headdim

    # Validate log-sum-exp (lse) shape and tensor strides
    assert headdim <= max_headdim
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert (
        q.is_contiguous()
        == k.is_contiguous()
        == v.is_contiguous()
        == output.is_contiguous()
        == True
    )
    assert dq.is_contiguous() == dk.is_contiguous() == dv.is_contiguous() == True

    # Set intermediate gradient buffers and softmax scale
    delta = torch.empty_like(lse)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)

    # Configure kernel launch parameters
    grid_preprocess = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK_Q_ROWS_SIZE"]),
        batch * nheads,
    )
    grid_main = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_KV_COLS_SIZE"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), min_block_headdim)

    # Launch Triton kernels with defined grids:
    #    Dim0 (kernel-specific):
    #    - For the pre-processing kernel blocks along query sequence
    #    - For the main backward kernel:
    #        * blocks along key sequence (if SEQUENCE_PARALLEL=True)
    #        * single (sequential) block processing (if SEQUENCE_PARALLEL=False)
    #    Dim1 (common for both):
    #    - Batch-head combinations
    _backward_preprocess_do_o_dot[grid_preprocess](
        output,
        d_output,
        delta,
        output.stride(batch_idx),
        output.stride(nheads_idx),
        output.stride(seqlen_idx),
        d_output.stride(batch_idx),
        d_output.stride(nheads_idx),
        d_output.stride(seqlen_idx),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        headdim,
        BLOCK_Q_ROWS_SIZE=block_rows_size,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )
    _backward_kernel(kernels_config)[grid_main](
        q,
        k,
        v,
        d_output,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(batch_idx),
        q.stride(nheads_idx),
        q.stride(seqlen_idx),
        k.stride(batch_idx),
        k.stride(nheads_idx),
        k.stride(seqlen_idx),
        v.stride(batch_idx),
        v.stride(nheads_idx),
        v.stride(seqlen_idx),
        d_output.stride(batch_idx),
        d_output.stride(nheads_idx),
        d_output.stride(seqlen_idx),
        dq_accum.stride(batch_idx),
        dq_accum.stride(nheads_idx),
        dq_accum.stride(seqlen_idx),
        dk.stride(batch_idx),
        dk.stride(nheads_idx),
        dk.stride(seqlen_idx),
        dv.stride(batch_idx),
        dv.stride(nheads_idx),
        dv.stride(seqlen_idx),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        seqlen_q // seqlen_cache_divisor,
        seqlen_k // seqlen_cache_divisor,
        BLOCK_HEADDIM,
    )

    dq.copy_(dq_accum)  # copy accumulated gradients into the final dq tensor


class FlashAttentionV2Function(torch.autograd.Function):  # type: ignore[misc]
    """Autograd function for FlashAttention v2."""

    @staticmethod
    def forward(
        ctx,  # noqa: ANN001
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        kernels_config: KernelsConfigV2,
    ) -> torch.Tensor:
        """Compute forward pass of FlashAttention-2.

        Optimized with Triton (attention outputs and intermediate values) needed for
        backward pass using tiling and kernel fusion techniques. Support fp16.

        Args:
            ctx: PyTorch autograd context (not used directly).
            q: Query tensor of shape (batch, nheads, seqlen_q, headdim).
            k: Key tensor of shape (batch, nheads, seqlen_k, headdim).
            v: Value tensor of shape (batch, nheads, seqlen_k, headdim).
            softmax_scale: Scaling factor for softmax (default: 1/sqrt(headdim)).
            kernels_config: Container for FlashAttention-2 Triton kernel parameters.

        Returns:
            output: Attention output tensor same shape as q.

        Note:
            Input data works only with contiguous and float16 (required condition).
            After calculations, the resulting output must be converted to contiguous
            and initial tensors dtype for numerical stability (required condition).
        """
        itinial_dtype = q.dtype

        # Convert tensors to contiguous and float16
        q, k, v = [proj.contiguous().to(torch.float16) for proj in (q, k, v)]

        # Compute forward pass with specified kernel parameters
        output, lse, ctx.softmax_scale = _flash_attention_forward(
            q,
            k,
            v,
            softmax_scale,
            kernels_config.block_rows_size,
            kernels_config.block_cols_size,
            kernels_config.min_block_headdim,
            kernels_config.max_headdim,
            kernels_config.seqlen_cache_divisor,
            kernels_config.min_warps,
            kernels_config.max_warps,
            kernels_config.num_stages,
        )

        # Save tensors and parameters for backward pass
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.kernels_config = kernels_config

        # Convert output to contiguous and itinial tensors dtype again
        return output.contiguous().to(itinial_dtype)

    @staticmethod
    def backward(
        ctx,  # noqa: ANN001
        d_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Compute gradients for query, key and value using FlashAttention-2 backward pass.

        Args:
            ctx: Context object from forward pass.
            d_output: Gradient of attention output (batch, nheads, seqlen_q, headdim).

        Returns:
            tuple: Gradients corresponding to forward arguments.

        Note:
            d_output is converted to float16 for internal gradient computation.
        """
        q, k, v, output, lse = ctx.saved_tensors
        kernels_config = ctx.kernels_config

        # Triton's autotuning modifies Tensor._version during kernel configuration, triggering
        # PyTorch's autograd safeguards that perform defensive copies (10-20% overhead).
        # Using inference_mode bypasses version tracking to eliminate this overhead
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)

            # Compute backward pass using parameters saved from forward
            _flash_attention_backward(
                q,
                k,
                v,
                output,
                lse,
                dq,
                dk,
                dv,
                d_output.to(torch.float16),  # required: kernels expect float16 gradients
                ctx.softmax_scale,
                kernels_config,
            )

        return dq, dk, dv, None, None


# === DEFAULT CONFIGURATION COMPONENTS WITHOUT BLACKWELL SUPPORT (TRITON 3.0.0-3.2.0) ===
# Optimal configurations for Turing
OLD_TURING_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": False},
        num_warps=4,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
OLD_TURING_KERNEL_CONFIG_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=128,
    block_cols_size=128,
    min_block_headdim=16,
    max_headdim=128,
    seqlen_cache_divisor=32,
    min_warps=4,
    max_warps=8,
    num_stages=1,
    backward_autotune_configs=OLD_TURING_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC,
)

OLD_TURING_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": True},
        num_warps=4,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
OLD_TURING_KERNEL_CONFIG_NON_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=128,
    block_cols_size=128,
    min_block_headdim=16,
    max_headdim=128,
    seqlen_cache_divisor=32,
    min_warps=4,
    max_warps=8,
    num_stages=1,
    backward_autotune_configs=OLD_TURING_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC,
)

# Optimal configurations for Ampere+ architectures
OLD_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 128, "BLOCK_KV_COLS_SIZE": 128, "SEQUENCE_PARALLEL": False},
        num_warps=8,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
OLD_AMPERE_PLUS_KERNEL_CONFIG_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=256,
    block_cols_size=256,
    min_block_headdim=32,
    max_headdim=256,
    seqlen_cache_divisor=32,
    min_warps=8,
    max_warps=16,
    num_stages=1,
    backward_autotune_configs=OLD_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC,
)

OLD_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 128, "BLOCK_KV_COLS_SIZE": 128, "SEQUENCE_PARALLEL": True},
        num_warps=8,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
OLD_AMPERE_PLUS_KERNEL_CONFIG_NON_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=256,
    block_cols_size=256,
    min_block_headdim=32,
    max_headdim=256,
    seqlen_cache_divisor=32,
    min_warps=8,
    max_warps=16,
    num_stages=1,
    backward_autotune_configs=OLD_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC,
)


# ==== DEFAULT CONFIGURATION COMPONENTS WITH BLACKWELL SUPPORT (TRITON 3.3.0 AND HIGHER) ====
# Optimal configurations for Turing
NEW_TURING_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 32, "BLOCK_KV_COLS_SIZE": 32, "SEQUENCE_PARALLEL": False},
        num_warps=4,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
NEW_TURING_KERNEL_CONFIG_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=128,
    block_cols_size=32,
    min_block_headdim=16,
    max_headdim=128,
    seqlen_cache_divisor=32,
    min_warps=4,
    max_warps=8,
    num_stages=1,
    backward_autotune_configs=NEW_TURING_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC,
)

NEW_TURING_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 32, "BLOCK_KV_COLS_SIZE": 32, "SEQUENCE_PARALLEL": True},
        num_warps=4,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
NEW_TURING_KERNEL_CONFIG_NON_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=128,
    block_cols_size=32,
    min_block_headdim=16,
    max_headdim=128,
    seqlen_cache_divisor=32,
    min_warps=4,
    max_warps=8,
    num_stages=1,
    backward_autotune_configs=NEW_TURING_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC,
)


# Optimal configurations for Ampere+ architectures
NEW_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": False},
        num_warps=8,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 128, "BLOCK_KV_COLS_SIZE": 128, "SEQUENCE_PARALLEL": False},
        num_warps=8,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
NEW_AMPERE_PLUS_KERNEL_CONFIG_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=256,
    block_cols_size=256,
    min_block_headdim=32,
    max_headdim=256,
    seqlen_cache_divisor=32,
    min_warps=8,
    max_warps=16,
    num_stages=1,
    backward_autotune_configs=NEW_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_DETERMINISTIC,
)

NEW_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC = [
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": True},
        num_warps=8,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
    triton.Config(
        {"BLOCK_Q_ROWS_SIZE": 128, "BLOCK_KV_COLS_SIZE": 128, "SEQUENCE_PARALLEL": True},
        num_warps=8,
        num_stages=1,
        pre_hook=init_to_zero_v2("DQ"),
    ),
]
NEW_AMPERE_PLUS_KERNEL_CONFIG_NON_DETERMINISTIC = KernelsConfigV2(
    block_rows_size=256,
    block_cols_size=256,
    min_block_headdim=32,
    max_headdim=256,
    seqlen_cache_divisor=32,
    min_warps=8,
    max_warps=16,
    num_stages=1,
    backward_autotune_configs=NEW_AMPERE_PLUS_BACKWARD_AUTOTUNE_CONFIGS_NON_DETERMINISTIC,
)


# Pre-tuned optimal, universal kernel configurations for different GPU architectures
# based on compute capability
_OLD_GPU_ARCHITECTURES = {
    "turing": {
        True: OLD_TURING_KERNEL_CONFIG_DETERMINISTIC,
        False: OLD_TURING_KERNEL_CONFIG_NON_DETERMINISTIC,
    },
    "ampere_plus": {
        True: OLD_AMPERE_PLUS_KERNEL_CONFIG_DETERMINISTIC,
        False: OLD_AMPERE_PLUS_KERNEL_CONFIG_NON_DETERMINISTIC,
    },
}
_NEW_GPU_ARCHITECTURES = {
    "turing": {
        True: NEW_TURING_KERNEL_CONFIG_DETERMINISTIC,
        False: NEW_TURING_KERNEL_CONFIG_NON_DETERMINISTIC,
    },
    "ampere_plus": {
        True: NEW_AMPERE_PLUS_KERNEL_CONFIG_DETERMINISTIC,
        False: NEW_AMPERE_PLUS_KERNEL_CONFIG_NON_DETERMINISTIC,
    },
}


# ========================= INTERNAL HELPERS FOR PUBLIC API =========================
def _validate_triton_compatibility(cc: tuple[int, int]) -> None:
    """Check the compatibility of the current Triton version with the current GPU architecture."""
    global _TURING_WARN_SHOWN

    if cc == _TURING_CC and _CURRENT_TRITON_VERSION > _OLD_TRITON_VERSION:
        if not _TURING_WARN_SHOWN:
            warnings.warn(
                f"Turing GPU detected with Triton {_CURRENT_TRITON_VERSION}. "
                "For optimal performance, use: pip install flash_attention_triton[legacy] "
                "or pip install triton==3.2.0\n"
                "Note: Blackwell GPUs will not be supported with the legacy version",
                UserWarning,
                stacklevel=2,
            )
            _TURING_WARN_SHOWN = True  # prevent repetitive alerts

    if cc >= _BLACKWELL_CC and _CURRENT_TRITON_VERSION < _NEW_TRITON_VERSION:
        raise RuntimeError(
            f"Blackwell GPU detected with Triton {_CURRENT_TRITON_VERSION}. "
            "Please use: pip install flash_attention_triton "
            "or pip install --upgrade triton>=3.3.0"
        )


def _select_optimal_gpu_architecture(cc: tuple[int, int], deterministic: bool) -> KernelsConfigV2:
    """Determine the appropriate GPU architecture configuration.

    Check the given compute capability (CC) and current Triton version, ensuring
    it meets minimum requirements (Turing and higher).

    Args:
        cc: GPU compute capability as a tuple (major, minor).
        deterministic: Flag for using the deterministic backward pass.

    Returns:
        Architecture configuration (Turing or Ampere+).

    Raises:
        ValueError: If the compute capability is below the minimum threshold.
    """
    if cc < _TURING_CC:
        raise ValueError(
            f"Unsupported GPU compute capability {cc}. Minimum required: {_TURING_CC} (Turing+)"
        )
    architecture = "turing" if cc == _TURING_CC else "ampere_plus"

    if _CURRENT_TRITON_VERSION < _NEW_TRITON_VERSION:
        return _OLD_GPU_ARCHITECTURES[architecture][deterministic]
    else:
        return _NEW_GPU_ARCHITECTURES[architecture][deterministic]


# =================================== PUBLIC API ===================================
def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Compute deterministic FlashAttention-2 with hardware-optimized kernels and causal masking.

    For heterogeneous GPU systems, each device uses its own optimal configuration.

    Automatically select pre-tuned optimal configuration based on GPU architecture:
        - Turing (CC 7.5):
            Turing (T4, RTX 20-series).
        - Ampere and above (CC 8.x+):
            Ampere (A100, RTX 30-series), Ada Lovelace (L40, RTX 40-series),
            Hopper (H100, H200), Blackwell (B100, B200, RTX 50-series), etc.

    Implementation Notes:
        Hardware requirements:
        - L1 cache ≥ 64KB per SM.
        - For older architectures or specialized tuning use flash_attention_v2_custom.

        Data Handling:
        - Input data will be automatically converted into contiguous and float16.
        - After calculations, the resulting output will be automatically converted
          to contiguous and initial input tensors dtype again for numerical stability.

    Args:
        q: Query tensor of shape (batch, nheads, seqlen_q, headdim).
        k: Key tensor of shape (batch, nheads, seqlen_k, headdim).
        v: Value tensor of shape (batch, nheads, seqlen_k, headdim).
        softmax_scale: Softmax scaling factor (default: 1/sqrt(headdim)).
        deterministic: Flag for using the deterministic backward pass, which is
            slightly slower and achieved by disabling sequence-parallel (atomic) operations.

    Returns:
        Attention output tensor same shape as q.
    """
    # Get gpu architecture config for the current device based on compute capability
    cc = torch.cuda.get_device_capability()
    _validate_triton_compatibility(cc)

    kernels_config = _select_optimal_gpu_architecture(cc, deterministic)

    return FlashAttentionV2Function.apply(  # type: ignore[no-any-return, no-untyped-call]
        q,
        k,
        v,
        softmax_scale,
        kernels_config,
    )


def flash_attention_v2_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    kernels_configs: dict[tuple[int, int], KernelsConfigV2],
) -> torch.Tensor:
    """Compute FlashAttention-2 with custom kernel configuration and causal masking.

    Support per-GPU configuration for heterogeneous systems.

    Args:
        q: Query tensor of shape (batch, nheads, seqlen_q, headdim).
        k: Key tensor of shape (batch, nheads, seqlen_k, headdim).
        v: Value tensor of shape (batch, nheads, seqlen_k, headdim).
        softmax_scale: Softmax scaling factor (if None, 1/sqrt(headdim) is used).
        kernels_configs: Dictionary mapping compute capability (major, minor)
            to KernelsConfig instances (requires L1 cache ≥ 64KB per SM).

    Returns:
        Attention output tensor same shape as q.

    Use case:
        Advanced performance tuning for each specific GPU architecture.

    Illustrative example:
        # Custom non-deterministic configuration for Turing GPUs
        turing_backward_autotune_config_non_deterministic = [
            triton.Config(
                {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": False},
                num_warps=4,
                num_stages=1,
                pre_hook=init_to_zero_v2("DQ"),
            ),
            triton.Config(
                {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": True},
                num_warps=4,
                num_stages=1,
                pre_hook=init_to_zero_v2("DQ"),
            ),
        ]
        turing_kernel_config_non_deterministic = KernelsConfigV2(
            block_rows_size=128,
            block_cols_size=128,
            min_block_headdim=16,
            max_headdim=128,
            seqlen_cache_divisor=32,
            min_warps=4,
            max_warps=8,
            num_stages=1,
            backward_autotune_configs=turing_backward_autotune_config_non_deterministic
        )

        # Create configuration mapping (several configs may be added here)
        non_deterministic_configs = {
            (7, 5): turing_kernel_config_non_deterministic  # Turing GPUs (T4, RTX 20-series)
        }

        # Compute attention with custom configurations
        output = flash_attention_v2_custom(
            q, k, v, softmax_scale=None, kernels_configs=non_deterministic_configs
        )

    Note:
        - Input data will be automatically converted into contiguous and float16.
        - After calculations, the resulting output will be automatically converted
            to contiguous and initial input tensors dtype again for numerical stability.
        - For correct results and stable behavior, it is recommended to use values >= 128
            for `block_rows_size`, `block_cols_size`, and `max_headdim`.
        - Non-determinism possible with custom configurations:
            1. Atomic operations in sequence-parallel mode (the main reason).
            2. Small block sizes (< 32) and extreme large num warps may increase risk.
    """
    cc = torch.cuda.get_device_capability()

    if cc not in kernels_configs:
        raise ValueError(
            f"Missing kernel configuration for compute capability {cc}. "
            "KernelsConfig must be provided for the current GPU architecture"
        )
    _validate_triton_compatibility(cc)

    kernels_config = kernels_configs[cc]

    return FlashAttentionV2Function.apply(  # type: ignore[no-any-return, no-untyped-call]
        q,
        k,
        v,
        softmax_scale,
        kernels_config,
    )
