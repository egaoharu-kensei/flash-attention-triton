![](https://github.com/egaoharu-kensei/fl-attn-triton/blob/master/assets/flash_attention_banner.png?raw=True)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-red)](https://www.python.org)
[![Triton Versions](https://img.shields.io/badge/Triton-3.1.0%2B-red)](https://triton-lang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0%2B-red)](https://pytorch.org)
[![PyPI version](https://badge.fury.io/py/flash-attention-triton.svg)](https://pypi.org/project/flash-attention-triton/)
[![Downloads](https://static.pepy.tech/badge/flash-attention-triton/month)](https://pepy.tech/projects/flash-attention-triton)

FlashAttention-2 Triton implementation based on Tri Dao's paper ["FlashAttention-2:
Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691).

## Key Features

- Cross-platform support (Linux and Windows)
- Dual-mode operation: deterministic (sequence-parallel disabled) and non-deterministic (higher performance)
- Hardware-aware optimizations for Turing (CC 7.5) and Ampere+ (CC 8.0+) architectures
- Custom configuration support for older GPU architectures or specialized tuning
- Support for homo and heterogeneous GPU clusters with automatic configuration selection

## Compatibility and Requirements

| Mode | GPU Architectures | PyTorch | CUDA | Triton | Python |
|------|-------------------|---------|------|--------|--------|
| **Legacy** | Turing-Hopper | 2.5.0-2.6.0 | 11.8+ | 3.1.0-3.2.0 | 3.10+ |
| **Modern** | Turing-Blackwell | 2.7.0+ | 12.8+ | 3.3.0+ | 3.10+ |

**Notes**

- Triton versions 3.3.0+ (at the moment 3.3.0-3.5.0) have issues (bugs) with increased shared memory usage 
on pre-Blackwell architectures (notably for Turing, reducing its performance to vanilla attention). For 
non-Blackwell GPUs (Turing-Hopper only) is recommended to use legacy mode.
- Microsoft Visual C++ Redistributable version 14.42 or higher is required for correct operation on Windows.
If you are using older versions, you must update the distributed components by downloading them from
the official [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) website, then copy the files msvcp140.dll, vcruntime140.dll and vcruntime140_1.dll
from the system directory C:\Windows\System32\ to the folder with an installed Python. In case of other
problems with Triton on Windows, it is recommended to review a solution in the [triton-windows](https://github.com/woct0rdho/triton-windows)
repository.

## Installation

First, install [PyTorch version with CUDA](https://pytorch.org/get-started/previous-versions/) depending on a Flash Attention mode you want to use.
This is a common requirement for both production and development environments.

### Production

Choose and perform one of the commands depending on GPU architecture:

```bash
pip install flash-attention-triton[legacy]
```

&emsp;or

```bash
pip install flash-attention-triton[modern]
```

### Development

1. Clone the repository and navigate into its directory: 

    ```bash
    git clone https://github.com/egaoharu-kensei/flash-attention-triton.git
    cd flash-attention-triton
    ```

2. Install the package in editable mode with development dependencies:

    ```bash
    pip install -e ".[dev,legacy]"
    ```

   &emsp;or

    ```bash
    pip install -e ".[dev,modern]"
    ```

3. Install pre-commit hooks once. Hooks will then run automatically on every git commit, checking all files:

    - First-time installation:
          
      ```bash
      pre-commit install
      ```

    - If pre-commit has already been installed and .pre-commit-config.yaml is updated, run:

      ```bash
      pre-commit install --overwrite
      ```

    - Optional: update hooks to the latest versions
  
      ```bash
      pre-commit autoupdate  
      ```

    - Optional: to run all checks on the entire project manually (e.g. after running tests or before committing):

      ```bash
      pre-commit run --all-files
      ```

4. Verify the installation by running the test suite:

    ```bash
    pytest -rs tests/ -v
    ```

**Note!** Using `pip install flash-attention-triton` or `pip install -e ".[dev]"` commands requires manual Triton installation.

## Api Documentation and Usage Examples

```python
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
```

```python
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
```

```python
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
```

```python
def init_to_zero_v2(name: str) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Pre-hook for triton.Config.

    Used in backward autotuning that initializes a tensor in nargs to zero by name.

    Args:
        name: Key identifying the tensor to be zero-initialized in the kernel arguments
                dictionary (e.g., "DQ" for query gradient, "DK" for key gradient).

    Returns:
        A function that zeros out the specified tensor inplace and then returns it.
    """
```

### Basic usage

Automatic kernel configuration

```python
import torch
from flash_attention_triton import flash_attention_v2


# Input tensors: (batch, n_heads, seq_len, head_dim)
q = torch.randn(16, 8, 512, 64, device="cuda")
k = torch.randn(16, 8, 512, 64, device="cuda")
v = torch.randn(16, 8, 512, 64, device="cuda")

# Automatic mode — hardware optimized
output = flash_attention_v2(q, k, v, softmax_scale=None, deterministic=True)
```

### Advanced customization with essential configuration components

Example for legacy mode

```python
import torch
import triton
from flash_attention_triton import KernelsConfigV2, flash_attention_v2_custom, init_to_zero_v2


# Input tensors: (batch, n_heads, seq_len, head_dim)
q = torch.randn(16, 8, 512, 64, device="cuda")
k = torch.randn(16, 8, 512, 64, device="cuda")
v = torch.randn(16, 8, 512, 64, device="cuda")

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

output = flash_attention_v2_custom(
    q, k, v, softmax_scale=None, kernels_configs=non_deterministic_configs
)
```

## Benchmarks

This section is under development and will be published as soon as it is possible to conduct a comprehensive 
comparative analysis of FlashAttention-2 for a wide range of GPU architectures in various tasks.
