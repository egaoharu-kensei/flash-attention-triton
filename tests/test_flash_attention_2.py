import functools
import gc

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from packaging import version

from flash_attention_triton import (
    KernelsConfigV2,
    flash_attention_v2,
    flash_attention_v2_custom,
    init_to_zero_v2,
)

_NEW_TRITON_VERSION = version.parse("3.3.0")
_CURRENT_TRITON_VERSION = version.parse(triton.__version__)

# Common set of parameters for all tests
IDS = ["small", "medium", "medium+", "large", "large+"]
STANDARD_SHAPES = [
    (8, 12, 512, 32),  # small size
    (16, 16, 1024, 48),  # medium size
    (32, 20, 1024, 64),  # medium+ size
    (32, 20, 1024, 96),  # large size
    (64, 32, 2048, 128),  # large+ size
]


# ================================= TEST HELPERS =================================
class AttentionModel(nn.Module):
    def forward(self, q, k, v, scale):
        return flash_attention_v2(q, k, v, scale)


def skip_resource_limits(func):
    """Decorator to skips tests when GPU resource limits are exceeded."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as oom_error:
            pytest.skip(str(oom_error))
        except triton.runtime.errors.OutOfResources as res_error:
            pytest.skip(str(res_error))
        except RuntimeError as rt_error:
            pytest.skip(str(rt_error))

    return wrapper


def generate_attention_tensors(batch, heads, seq_len, d_head):
    """Generate input tensors for attention tests."""
    device = "cuda"
    dtype = torch.float32

    q = torch.randn(batch, heads, seq_len, d_head, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, d_head, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, d_head, dtype=dtype, device=device, requires_grad=True)

    return {
        "q": q,
        "k": k,
        "v": v,
        "sm_scale": 1.0 / (d_head**0.5),
        "expected_shape": (batch, heads, seq_len, d_head),
        "seq_len": seq_len,
        "device": device,
    }


@pytest.fixture(scope="function", params=STANDARD_SHAPES, ids=IDS)
def attention_inputs(request):
    def _generate():
        return generate_attention_tensors(*request.param)

    return _generate


def reset_environment():
    """Complete environment reset with the fixed seed."""
    # Memory clearing
    gc.collect()
    torch.cuda.empty_cache()

    # Reset the random number generator
    torch.manual_seed(42)


def run_forward_backward(inputs, attn_func, **attn_func_kwargs):
    """Perform one forward and backward passes.

    Args:
        inputs: Dictionary with 'q', 'k', 'v', 'sm_scale'.
        attn_func: Attention function to test.
        **attn_func_kwargs: Additional attention function-specific arguments.

    Returns:
        Dictionary with output and gradients.
    """
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    sm_scale = inputs["sm_scale"]

    grad_output = torch.randn_like(q)

    # Forward and backward passes
    output = attn_func(q, k, v, sm_scale, **attn_func_kwargs)
    output.backward(grad_output)

    return {
        "output": output.detach().clone(),
        "dq": q.grad.detach().clone(),
        "dk": k.grad.detach().clone(),
        "dv": v.grad.detach().clone(),
    }


def create_custom_kernels_config():
    """Create universal configuration mapping based on cc of the current device."""
    # The same kernels as in flash_attention_v2 by default depending on Triton versions
    if _CURRENT_TRITON_VERSION < _NEW_TRITON_VERSION:
        # Turing
        turing_backward_autotune_config_deterministic = [
            triton.Config(
                {"BLOCK_Q_ROWS_SIZE": 64, "BLOCK_KV_COLS_SIZE": 64, "SEQUENCE_PARALLEL": False},
                num_warps=4,
                num_stages=1,
                pre_hook=init_to_zero_v2("DQ"),
            ),
        ]
        turing_kernel_config_deterministic = KernelsConfigV2(
            block_rows_size=128,
            block_cols_size=128,
            min_block_headdim=16,
            max_headdim=128,
            seqlen_cache_divisor=32,
            min_warps=4,
            max_warps=8,
            num_stages=1,
            backward_autotune_configs=turing_backward_autotune_config_deterministic,
        )

        # Ampere+
        ampere_plus_backward_autotune_configs_deterministic = [
            triton.Config(
                {"BLOCK_Q_ROWS_SIZE": 128, "BLOCK_KV_COLS_SIZE": 128, "SEQUENCE_PARALLEL": False},
                num_warps=8,
                num_stages=1,
                pre_hook=init_to_zero_v2("DQ"),
            ),
        ]
        ampere_plus_config_deterministic = KernelsConfigV2(
            block_rows_size=256,
            block_cols_size=256,
            min_block_headdim=32,
            max_headdim=256,
            seqlen_cache_divisor=32,
            min_warps=8,
            max_warps=16,
            num_stages=1,
            backward_autotune_configs=ampere_plus_backward_autotune_configs_deterministic,
        )
    else:
        # Turing
        turing_backward_autotune_config_deterministic = [
            triton.Config(
                {"BLOCK_Q_ROWS_SIZE": 32, "BLOCK_KV_COLS_SIZE": 32, "SEQUENCE_PARALLEL": False},
                num_warps=4,
                num_stages=1,
                pre_hook=init_to_zero_v2("DQ"),
            ),
        ]
        turing_kernel_config_deterministic = KernelsConfigV2(
            block_rows_size=128,
            block_cols_size=32,
            min_block_headdim=16,
            max_headdim=128,
            seqlen_cache_divisor=32,
            min_warps=4,
            max_warps=8,
            num_stages=1,
            backward_autotune_configs=turing_backward_autotune_config_deterministic,
        )

        # Ampere+
        ampere_plus_backward_autotune_configs_deterministic = [
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
        ampere_plus_config_deterministic = KernelsConfigV2(
            block_rows_size=256,
            block_cols_size=256,
            min_block_headdim=32,
            max_headdim=256,
            seqlen_cache_divisor=32,
            min_warps=8,
            max_warps=16,
            num_stages=1,
            backward_autotune_configs=ampere_plus_backward_autotune_configs_deterministic,
        )

    # Compute capability for the current device
    cc = torch.cuda.get_device_capability()

    # Skip test if GPU is unsupported
    if cc < (7, 5):
        pytest.skip(f"Unsupported GPU compute capability: {cc}")
    elif cc == (7, 5):
        kernel_config = turing_kernel_config_deterministic
    else:
        kernel_config = ampere_plus_config_deterministic

    return {cc: kernel_config}


# ================================== MAIN TESTS ==================================
def test_cuda_is_available():
    assert torch.cuda.is_available() == True, "Only CUDA devices are available"


@skip_resource_limits
def test_output_shape(attention_inputs):
    """Check if output tensor has correct dimensions."""
    inputs = attention_inputs()
    output = flash_attention_v2(inputs["q"], inputs["k"], inputs["v"], inputs["sm_scale"])

    assert output.shape == inputs["expected_shape"], (
        f"Output shape {output.shape} doesn't match expected {inputs['expected_shape']}"
    )


@skip_resource_limits
def test_no_nan_output(attention_inputs):
    """Test if output tensor doesn't contain NaN values."""
    inputs = attention_inputs()
    output = flash_attention_v2(inputs["q"], inputs["k"], inputs["v"], inputs["sm_scale"])

    assert not torch.isnan(output).any(), "Output contains NaN values"


@skip_resource_limits
def test_no_nan_gradients(attention_inputs):
    """Ensure gradients don't contain NaN values."""
    inputs = attention_inputs()
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    sm_scale = inputs["sm_scale"]

    output = flash_attention_v2(q, k, v, sm_scale)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    for param, name in zip([q, k, v], ["q", "k", "v"], strict=False):
        assert not torch.isnan(param.grad).any(), f"Gradients contain NaN values in {name}"


@skip_resource_limits
def test_compare_vanilla_attention(attention_inputs):
    """Compare FlashAttention v2 with Vanilla implementation."""
    inputs = attention_inputs()
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    sm_scale = inputs["sm_scale"]
    seq_len = inputs["seq_len"]
    device = inputs["device"]

    # Compute FlashAttention v2 output
    output_flash = flash_attention_v2(q, k, v, sm_scale)

    # Compute Vanilla Attention output
    mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1
    )
    qk = (q @ k.transpose(-2, -1)) * sm_scale
    qk = qk.masked_fill(mask, float("-inf"))
    output_vanilla = torch.softmax(qk, dim=-1) @ v

    # Compare results
    assert torch.allclose(output_flash, output_vanilla, atol=1e-2, rtol=1e-3), (
        "FlashAttention-2 output doesn't match Vanilla Attention implementation"
    )


@skip_resource_limits
def test_compare_torch_attention(attention_inputs):
    """Compare FlashAttention v2 with PyTorch V2 implementation."""
    inputs = attention_inputs()
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    sm_scale = inputs["sm_scale"]

    # Compute FlashAttention outputs
    output_flash = flash_attention_v2(q, k, v, sm_scale)
    output_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

    # Compare results
    assert torch.allclose(output_flash, output_torch, atol=1e-2, rtol=1e-3), (
        "FlashAttention-2 output doesn't match PyTorch implementation"
    )


@skip_resource_limits
def test_determinism(attention_inputs, num_iters=100):
    """Testing deterministic mode with a fixed seed (42)."""
    check_keys = ["output", "dq", "dk", "dv"]

    # First run (reference)
    reset_environment()
    ref_inputs = attention_inputs()
    reference = run_forward_backward(ref_inputs, flash_attention_v2, deterministic=True)

    # Further runs
    for i in range(1, num_iters + 1):
        reset_environment()
        iter_inputs = attention_inputs()
        results = run_forward_backward(iter_inputs, flash_attention_v2, deterministic=True)

        # Check the match with the reference
        for key in check_keys:
            assert torch.equal(results[key], reference[key]), (
                f"Non-determinism detected in deterministic mode. "
                f"Mismatch at iteration {i} for {key}"
            )


@skip_resource_limits
def test_non_determinism(attention_inputs, num_iters=100):
    """Testing non-deterministic mode with a fixed seed (42).

    Note:
        Due to atomic operations on small-dimensional data, non-determinism
        may not always manifest itself, so it may require more iterations
        for testing and even several restarts of this test (non-critical
        and may be skipped if deterministic behavior is observed).

    """
    check_keys = ["output", "dq", "dk", "dv"]
    non_determinism = False

    # First run (reference)
    reset_environment()
    ref_inputs = attention_inputs()
    reference = run_forward_backward(ref_inputs, flash_attention_v2, deterministic=False)

    # Further runs
    for _ in range(num_iters):
        reset_environment()

        iter_inputs = attention_inputs()
        results = run_forward_backward(iter_inputs, flash_attention_v2, deterministic=False)

        # If there is any difference between current results and reference
        if any(not torch.equal(results[key], reference[key]) for key in check_keys):
            non_determinism = True
            break

    # Assert non-determinism was detected or skip if not
    if non_determinism:
        assert non_determinism
    else:
        pytest.skip(
            "Determinism detected in non-deterministic mode.\n"
            "This is an expected outcome in some environments due to hardware-level "
            "consistency in atomic operation execution and does not indicate "
            "implementation defects."
        )


@skip_resource_limits
def test_flash_attention_v2_custom(attention_inputs):
    """Ensure correctness of custom FlashAttention-2 kernel configuration.

    Compare outputs and gradients between the custom kernel configuration
    and default deterministic implementation using identical parameters.
    """
    check_keys = ["output", "dq", "dk", "dv"]

    reset_environment()
    std_inputs = attention_inputs()
    output_flash = run_forward_backward(std_inputs, flash_attention_v2, deterministic=True)

    reset_environment()
    custom_inputs = attention_inputs()
    det_kernels_config = create_custom_kernels_config()
    output_flash_custom = run_forward_backward(
        custom_inputs, flash_attention_v2_custom, kernels_configs=det_kernels_config
    )

    for key in check_keys:
        assert torch.equal(output_flash[key], output_flash_custom[key]), (
            f"Results mismatch for {key} between default and custom implementations"
        )


@skip_resource_limits
def test_flash_attention_data_parallel(attention_inputs):
    """Test FlashAttention operation under nn.DataParallel.

    Needed to ensure consistent behavior across multiple GPUs compared to single GPU execution.

    Note:
        - Require minimum 2 GPUs but can work with 1 to avoid execution error.
        - Batch size must be divisible by number of GPUs.
        - All tensors must reside on CUDA device 0.
    """
    # Determine available GPUs
    main_device = torch.device("cuda:0")
    num_gpus = torch.cuda.device_count()
    # assert num_gpus > 1, "Test requires at least 2 GPUs"

    # Generate base inputs and move to primary GPU
    inputs = attention_inputs()
    q = inputs["q"].to(main_device)
    k = inputs["k"].to(main_device)
    v = inputs["v"].to(main_device)
    sm_scale = inputs["sm_scale"]

    # Prepare DataParallel model
    model_dp = nn.DataParallel(AttentionModel(), device_ids=list(range(num_gpus)))
    model_dp.to(main_device)  # Primary model on cuda:0

    # ==================== DataParallel execution ====================
    reset_environment()
    q_parallel = q.detach().clone().requires_grad_(True)
    k_parallel = k.detach().clone().requires_grad_(True)
    v_parallel = v.detach().clone().requires_grad_(True)

    # Forward pass (automatically splits batch across GPUs)
    output_parallel = model_dp(q_parallel, k_parallel, v_parallel, sm_scale)

    # Backward pass (gradients aggregated on the main GPU)
    output_parallel.backward(gradient=torch.randn_like(q_parallel))

    # ==================== Single GPU execution ====================
    reset_environment()
    q_single = q.detach().clone().requires_grad_(True)
    k_single = k.detach().clone().requires_grad_(True)
    v_single = v.detach().clone().requires_grad_(True)

    # Direct function call
    output_single = flash_attention_v2(q_single, k_single, v_single, sm_scale)

    # Backward pass
    output_single.backward(gradient=torch.randn_like(q_single))

    # ==================== Validation checks ====================
    atol, rtol = 1e-3, 1e-3

    # Output and gradient tensors comparison
    assert torch.allclose(output_parallel, output_single, atol=atol, rtol=rtol), (
        "Outputs mismatch between parallel and single GPU"
    )
    assert torch.allclose(q_parallel.grad, q_single.grad, atol=atol, rtol=rtol), (
        "Q gradients mismatch between parallel and single GPU"
    )
    assert torch.allclose(k_parallel.grad, k_single.grad, atol=atol, rtol=rtol), (
        "K gradients mismatch between parallel and single GPU"
    )
    assert torch.allclose(v_parallel.grad, v_single.grad, atol=atol, rtol=rtol), (
        "V gradients mismatch between parallel and single GPU"
    )
