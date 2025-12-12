import triton
import triton.language as tl

from ._configurators_v2 import KernelsConfigV2

# Cache for compiled backward kernels to avoid recompilation
_BACKWARD_KERNEL_CACHE = {}  # type: ignore[var-annotated]


# ==================================== CORE LOGIC ====================================
@triton.jit
def _forward_kernel(
    Q,
    K,
    V,
    Output,
    LSE,
    ScratchpadBuffer,
    softmax_scale,
    stride_q_batch,
    stride_q_head,
    stride_q_seqlen,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_output_batch,
    stride_output_head,
    stride_output_seqlen,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_Q_ROWS_SIZE: tl.constexpr,
    BLOCK_KV_COLS_SIZE: tl.constexpr,
):
    """Triton JIT-compiled kernel implementation of FlashAttention-2 forward pass.

    Compute attention outputs and log-sum-exp values for numerical stability in batched
    multi-head attention with causal masking. Optimized for GPU execution
    using block-wise computations and shared memory strategies.

    Args:
        Q: Input query tensor pointer of shape (batch, nheads, seqlen_q, headdim).
        K: Input key tensor pointer of shape (batch, nheads, seqlen_k, headdim).
        V: Input value tensor pointer of shape (batch, nheads, seqlen_k, headdim).
        Output: Output tensor pointer of shape (batch, nheads, seqlen_q, headdim).
        LSE: Log-sum-exp buffer pointer for backward pass.
        ScratchpadBuffer: Temporary buffer for normalization factors.
        softmax_scale: Scaling factor for softmax.
        stride_*: Strides in elements for corresponding tensor dimensions.
        nheads: Number of attention heads.
        seqlen_q: Sequence length of queries.
        seqlen_k: Sequence length of keys/values.
        headdim: Dimension size of each attention head.
        BLOCK_HEADDIM: Block size for head dimension (constexpr).
        BLOCK_Q_ROWS_SIZE: Block size for processing query rows (constexpr).
            Controls tiling along sequence dimension for Q.
        BLOCK_KV_COLS_SIZE: Block size for processing key/value columns (constexpr).
            Controls tiling along sequence dimension for K/V.

    Notes:
        1. Uses block-wise matrix multiplication with double buffering.
        2. Maintains numerical stability through online max-value tracking.
        3. Implements FlashAttention-2 algorithm with fused kernel design.
        4. Requires BLOCK_Q_ROWS_SIZE and BLOCK_KV_COLS_SIZE as powers of 2.
    """
    # Determine Q and K/V block boundaries
    start_q_rows_block = tl.program_id(0)
    end_kv_cols_block = tl.minimum((start_q_rows_block + 1) * BLOCK_Q_ROWS_SIZE, seqlen_k)

    # Initialize offsets
    offset_batch_head = tl.program_id(1)
    offset_batch = offset_batch_head // nheads
    offset_head = offset_batch_head % nheads

    offsets_q_rows_block = start_q_rows_block * BLOCK_Q_ROWS_SIZE + tl.arange(0, BLOCK_Q_ROWS_SIZE)
    offsets_kv_cols_inblock = tl.arange(0, BLOCK_KV_COLS_SIZE)
    offsets_headdim = tl.arange(0, BLOCK_HEADDIM)

    # Compute tensor pointers for Q, K, V
    q_pointers = (
        Q
        + offset_batch * stride_q_batch
        + offset_head * stride_q_head
        + (offsets_q_rows_block[:, None] * stride_q_seqlen + offsets_headdim[None, :])
    )
    k_pointers = (
        K
        + offset_batch * stride_k_batch
        + offset_head * stride_k_head
        + (offsets_kv_cols_inblock[:, None] * stride_k_seqlen + offsets_headdim[None, :])
    )
    v_pointers = (
        V
        + offset_batch * stride_v_batch
        + offset_head * stride_v_head
        + (offsets_kv_cols_inblock[:, None] * stride_v_seqlen + offsets_headdim[None, :])
    )
    scratchpad_pointers = (
        ScratchpadBuffer + offset_batch_head * seqlen_q_rounded + offsets_q_rows_block
    )

    # Initialize accumulators
    block_lse_accum = tl.zeros([BLOCK_Q_ROWS_SIZE], dtype=tl.float32) - float("inf")
    block_max_qk_accum = tl.zeros([BLOCK_Q_ROWS_SIZE], dtype=tl.float32) - float("inf")
    output_accum = tl.zeros([BLOCK_Q_ROWS_SIZE, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q block with mask for boundaries check
    q_block_mask = (offsets_q_rows_block[:, None] < seqlen_q) & (
        offsets_headdim[None, :] < headdim
    )
    q_block = tl.load(q_pointers, mask=q_block_mask, other=0.0)

    # Process K/V blocks aligned relative to the current Q block boundaries
    for start_kv_cols_block in range(0, end_kv_cols_block, BLOCK_KV_COLS_SIZE):
        start_kv_cols_block = tl.multiple_of(start_kv_cols_block, BLOCK_KV_COLS_SIZE)

        # Load K/V with mask for boundaries check
        kv_block_mask = ((start_kv_cols_block + offsets_kv_cols_inblock)[:, None] < seqlen_k) & (
            offsets_headdim[None, :] < headdim
        )
        k_block = tl.load(
            k_pointers + start_kv_cols_block * stride_k_seqlen, mask=kv_block_mask, other=0.0
        )
        v_block = tl.load(
            v_pointers + start_kv_cols_block * stride_v_seqlen, mask=kv_block_mask, other=0.0
        )

        # Compute attention scores with causal masking
        qk_block = tl.zeros([BLOCK_Q_ROWS_SIZE, BLOCK_KV_COLS_SIZE], dtype=tl.float32)
        qk_block += tl.dot(q_block, tl.trans(k_block))
        qk_block += tl.where(
            (start_kv_cols_block + offsets_kv_cols_inblock)[None, :] < seqlen_k, 0, float("-inf")
        )
        qk_block += tl.where(
            offsets_q_rows_block[:, None]
            >= (start_kv_cols_block + offsets_kv_cols_inblock)[None, :],
            0,
            float("-inf"),
        )

        # Stable online softmax
        new_block_max_qk = tl.maximum(tl.max(qk_block * softmax_scale, 1), block_lse_accum)
        exp_scores = tl.exp(qk_block * softmax_scale - new_block_max_qk[:, None])
        rowsum_exp_scores = tl.sum(exp_scores, 1)

        # Normalize accumulated output from previous blocks relative to the new maximum
        # and add new weighted values
        output_accum_norm_factor = tl.exp(block_max_qk_accum - new_block_max_qk)
        tl.store(scratchpad_pointers, output_accum_norm_factor)
        output_accum_norm_factor = tl.load(scratchpad_pointers)  # bypass register restrictions
        output_accum = output_accum * output_accum_norm_factor[:, None]
        output_accum += tl.dot(exp_scores.to(v_block.dtype), v_block)

        # Update max_qk and lse accumulators
        block_max_qk_accum = new_block_max_qk
        new_block_lse_accum = tl.exp(block_lse_accum - new_block_max_qk) + rowsum_exp_scores
        block_lse_accum = new_block_max_qk + tl.log(new_block_lse_accum)

    # Final output accumulators normalization
    output_norm_factor = tl.exp(block_max_qk_accum - block_lse_accum)
    tl.store(scratchpad_pointers, output_norm_factor)
    output_norm_factor = tl.load(scratchpad_pointers)
    output_accum = output_accum * output_norm_factor[:, None]

    # Rematerialize offsets to save registers
    start_q_rows_block = tl.program_id(0)
    offsets_q_rows_block = start_q_rows_block * BLOCK_Q_ROWS_SIZE + tl.arange(0, BLOCK_Q_ROWS_SIZE)

    # Store lse results
    lse_pointers = LSE + offset_batch_head * seqlen_q_rounded + offsets_q_rows_block
    tl.store(lse_pointers, block_lse_accum)

    # Store output results
    offsets_headdim = tl.arange(0, BLOCK_HEADDIM)
    output_pointers = (
        Output
        + offset_batch * stride_output_batch
        + offset_head * stride_output_head
        + (offsets_q_rows_block[:, None] * stride_output_seqlen + offsets_headdim[None, :])
    )
    tl.store(output_pointers, output_accum, mask=q_block_mask)


@triton.jit
def _backward_kernel_one_col_block(
    start_kv_cols_block,
    Q,
    K,
    V,
    DOutput,
    DQ,
    DK,
    DV,
    LSE,
    Delta,
    softmax_scale,
    stride_q_seqlen,
    stride_k_seqlen,
    stride_v_seqlen,
    stride_d_output_seqlen,
    stride_dq_seqlen,
    stride_dk_seqlen,
    stride_dv_seqlen,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_Q_ROWS_SIZE: tl.constexpr,
    BLOCK_KV_COLS_SIZE: tl.constexpr,
):
    """Process a single column block of K/V in backward pass of FlashAttention-2.

    Compute gradients for keys, values and queries through block-wise matrix
    operations with online softmax correction and optional atomic updates.

    Args:
        start_kv_cols_block: Starting position of K/V column block.
        Q: Input query tensor pointer of shape (batch, nheads, seqlen_q, headdim).
        K: Input key tensor pointer of shape (batch, nheads, seqlen_k, headdim).
        V: Input value tensor pointer of shape (batch, nheads, seqlen_k, headdim).
        DOutput: Gradient of output tensor pointer of shape (batch, nheads, seqlen_q, headdim).
        DQ: Gradient accumulator for queries.
        DK: Gradient accumulator for keys.
        DV: Gradient accumulator for values.
        LSE: Log-sum-exp values from forward pass.
        Delta: Precomputed delta values.
        softmax_scale: Scaling factor for softmax.
        stride_*_seqlen: Strides for sequence dimension.
        seqlen_q: Sequence length of queries.
        seqlen_k: Sequence length of keys/values.
        headdim: Dimension size of each attention head.
        ATOMIC_ADD: Flag for atomic update strategy.
        BLOCK_HEADDIM: Block size for head dimension (constexpr).
        BLOCK_Q_ROWS_SIZE: Block size for processing query rows (constexpr).
            Controls tiling along sequence dimension for Q.
        BLOCK_KV_COLS_SIZE: Block size for processing key/value columns (constexpr).
            Controls tiling along sequence dimension for K/V.

    Key Operations:
        1. Loads and processes K/V block once.
        2. Iterates over all Q blocks in causal dependency range.
        3. Computes attention gradients via chain rule.
        4. Accumulates gradients with numerical stability.
        5. Handles boundary conditions via masking.
    """
    # Determine aligned Q block boundaries
    aligned_start_q_rows_block = (
        (start_kv_cols_block * BLOCK_KV_COLS_SIZE) // BLOCK_Q_ROWS_SIZE
    ) * BLOCK_Q_ROWS_SIZE

    # Initialize offsets
    offsets_q_rows_block = aligned_start_q_rows_block + tl.arange(0, BLOCK_Q_ROWS_SIZE)
    offsets_q_rows_inblock = tl.arange(0, BLOCK_Q_ROWS_SIZE)
    offsets_kv_cols_block = start_kv_cols_block * BLOCK_KV_COLS_SIZE + tl.arange(
        0, BLOCK_KV_COLS_SIZE
    )
    offsets_headdim = tl.arange(0, BLOCK_HEADDIM)

    # Compute tensor pointers for Q, K, V and for their gradients
    q_pointers = Q + (offsets_q_rows_block[:, None] * stride_q_seqlen + offsets_headdim[None, :])
    k_pointers = K + (offsets_kv_cols_block[:, None] * stride_k_seqlen + offsets_headdim[None, :])
    v_pointers = V + (offsets_kv_cols_block[:, None] * stride_v_seqlen + offsets_headdim[None, :])

    dq_pointers = DQ + (
        offsets_q_rows_block[:, None] * stride_dq_seqlen + offsets_headdim[None, :]
    )
    dk_pointers = DK + (
        offsets_kv_cols_block[:, None] * stride_dk_seqlen + offsets_headdim[None, :]
    )
    dv_pointers = DV + (
        offsets_kv_cols_block[:, None] * stride_dv_seqlen + offsets_headdim[None, :]
    )
    d_output_pointers = DOutput + (
        offsets_q_rows_block[:, None] * stride_d_output_seqlen + offsets_headdim[None, :]
    )

    # Initialize gradient accumulators
    dk_accum = tl.zeros([BLOCK_KV_COLS_SIZE, BLOCK_HEADDIM], dtype=tl.float32)
    dv_accum = tl.zeros([BLOCK_KV_COLS_SIZE, BLOCK_HEADDIM], dtype=tl.float32)

    # Load K/V blocks with mask for boundaries check
    kv_block_mask = (offsets_kv_cols_block[:, None] < seqlen_k) & (
        offsets_headdim[None, :] < headdim
    )
    k_block = tl.load(k_pointers, mask=kv_block_mask, other=0.0)
    v_block = tl.load(v_pointers, mask=kv_block_mask, other=0.0)

    # Process Q blocks aligned relative to the current K/V block boundaries
    num_q_rows_blocks = tl.cdiv(seqlen_q, BLOCK_Q_ROWS_SIZE)
    for start_q_rows_block in range(
        aligned_start_q_rows_block, num_q_rows_blocks * BLOCK_Q_ROWS_SIZE, BLOCK_Q_ROWS_SIZE
    ):
        offsets_current_q_rows_block = start_q_rows_block + offsets_q_rows_inblock

        # Load Q block with mask for boundaries check
        # (this mask is also the same for d_output and d_q)
        q_block_mask = (offsets_current_q_rows_block[:, None] < seqlen_q) & (
            offsets_headdim[None, :] < headdim
        )
        q_block = tl.load(q_pointers, mask=q_block_mask, other=0.0)

        # Compute attention scores with causal masking
        qk_block = tl.dot(q_block, tl.trans(k_block))
        qk_block = tl.where(offsets_kv_cols_block[None, :] < seqlen_k, qk_block, float("-inf"))
        qk_block = tl.where(
            offsets_current_q_rows_block[:, None] >= (offsets_kv_cols_block[None, :]),
            qk_block,
            float("-inf"),
        )

        # Stable online softmax
        block_lse = tl.load(LSE + offsets_current_q_rows_block)
        exp_scores = tl.exp(qk_block * softmax_scale - block_lse[:, None])

        # Load d_output with mask and d_delta
        d_output = tl.load(d_output_pointers, mask=q_block_mask, other=0.0)
        d_delta = tl.load(Delta + offsets_current_q_rows_block)

        # Compute and accumulate gradients
        d_exp_scores = tl.dot(d_output, tl.trans(v_block))
        d_softmax_logits = (exp_scores * (d_exp_scores - d_delta[:, None]) * softmax_scale).to(
            q_block.dtype
        )
        dk_accum += tl.dot(tl.trans(d_softmax_logits), q_block)
        dv_accum += tl.dot(tl.trans(exp_scores.to(d_output.dtype)), d_output)

        # Update DQ depending on ATOMIC_ADD condition
        if ATOMIC_ADD:
            dq = tl.dot(d_softmax_logits, k_block)
            tl.atomic_add(dq_pointers, dq, mask=q_block_mask)
        else:
            dq = tl.load(dq_pointers, mask=q_block_mask, other=0.0)
            dq += tl.dot(d_softmax_logits, k_block)
            tl.store(dq_pointers, dq, mask=q_block_mask)

        # Shift pointers
        q_pointers += BLOCK_Q_ROWS_SIZE * stride_q_seqlen
        dq_pointers += BLOCK_Q_ROWS_SIZE * stride_dq_seqlen
        d_output_pointers += BLOCK_Q_ROWS_SIZE * stride_d_output_seqlen

    # Store dk, dv results
    tl.store(dk_pointers, dk_accum, mask=kv_block_mask)
    tl.store(dv_pointers, dv_accum, mask=kv_block_mask)


def _backward_kernel(kernels_config: KernelsConfigV2):
    """Create or retrieve a cached Triton kernel for FlashAttention-2 backward pass.

    Use GPU-specific configuration and autotuning to avoid recompilation overhead.
    """
    config_key = kernels_config._create_kernel_key()

    if config_key in _BACKWARD_KERNEL_CACHE:
        return _BACKWARD_KERNEL_CACHE[config_key]

    @triton.autotune(
        configs=kernels_config.backward_autotune_configs,
        key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BLOCK_HEADDIM"],
    )
    @triton.jit
    def _create_backward_kernel(
        Q,
        K,
        V,
        DOutput,
        DQ,
        DK,
        DV,
        LSE,
        Delta,
        softmax_scale,
        stride_q_batch,
        stride_q_head,
        stride_q_seqlen,
        stride_k_batch,
        stride_k_head,
        stride_k_seqlen,
        stride_v_batch,
        stride_v_head,
        stride_v_seqlen,
        stride_d_output_batch,
        stride_d_output_head,
        stride_d_output_seqlen,
        stride_dq_batch,
        stride_dq_head,
        stride_dq_seqlen,
        stride_dk_batch,
        stride_dk_head,
        stride_dk_seqlen,
        stride_dv_batch,
        stride_dv_head,
        stride_dv_seqlen,
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM: tl.constexpr,
        SEQUENCE_PARALLEL: tl.constexpr,
        BLOCK_Q_ROWS_SIZE: tl.constexpr,
        BLOCK_KV_COLS_SIZE: tl.constexpr,
    ):
        """Triton kernel for backward pass of FlashAttention-2 (gradient computation).

        Compute gradients for Q, K, V tensors based on the output gradient (DOutput)
        and attention intermediates (LSE, Delta). Support sequence-parallel
        and non-parallel execution modes.

        Key Features:
            - Autotuned for optimal block sizes (BLOCK_Q_ROWS_SIZE, BLOCK_KV_COLS_SIZE).
            - Double-buffering and tiling for memory efficiency.
            - Sequence-parallel processing option for long sequences.

        Args:
            Q, K, V: Input tensors of shapes (batch, nheads, seqlen, headdim).
            DOutput: Gradient of output tensor of shape same as Q.
            DQ, DK, DV: Output gradient tensors (to be computed) of shape same as Q/K/V.
            LSE: Log-sum-exp values from forward pass of shape (batch, nheads, seqlen_q)
               interpreted as 2D tensor (batch*nheads, seqlen_q_rounded) in a kernel.
            Delta: Intermediate tensor with same shape and interpretation as LSE.
            softmax_scale: Scaling factor for softmax.
            stride_*_batch/head/seqlen: Stride parameters for tensor memory layout.
            nheads: Number of attention heads.
            seqlen_q: Sequence length of queries.
            seqlen_k: Sequence length of keys/values.
            seqlen_q_rounded: Sequence length of queries rounded to block size.
            headdim: Dimension size of each attention head.
            CACHE_KEY_SEQLEN_Q/K: Autotuning keys (ignored during execution).
            BLOCK_HEADDIM: Block size for head dimension (constexpr).
            SEQUENCE_PARALLEL: Flag to use sequence-parallel processing.
            BLOCK_Q_ROWS_SIZE: Block size for processing query rows (constexpr).
                Controls tiling along sequence dimension for Q.
            BLOCK_KV_COLS_SIZE: Block size for processing key/value columns (constexpr).
                Controls tiling along sequence dimension for K/V.
        """
        # Initialize offsets
        offset_batch_head = tl.program_id(1)
        offset_batch = offset_batch_head // nheads
        offset_head = offset_batch_head % nheads

        # Adjust offset pointers for current batch-head and for their gradients
        Q += offset_batch * stride_q_batch + offset_head * stride_q_head
        K += offset_batch * stride_k_batch + offset_head * stride_k_head
        V += offset_batch * stride_v_batch + offset_head * stride_v_head

        DOutput += offset_batch * stride_d_output_batch + offset_head * stride_d_output_head
        DQ += offset_batch * stride_dq_batch + offset_head * stride_dq_head
        DK += offset_batch * stride_dk_batch + offset_head * stride_dk_head
        DV += offset_batch * stride_dv_batch + offset_head * stride_dv_head

        # Pointers to row-wise intermediates (LSE, Delta)
        Delta += offset_batch_head * seqlen_q_rounded
        LSE += offset_batch_head * seqlen_q_rounded

        if SEQUENCE_PARALLEL:
            start_kv_cols_block = tl.program_id(0)
            _backward_kernel_one_col_block(
                start_kv_cols_block,
                Q,
                K,
                V,
                DOutput,
                DQ,
                DK,
                DV,
                LSE,
                Delta,
                softmax_scale,
                stride_q_seqlen,
                stride_k_seqlen,
                stride_v_seqlen,
                stride_d_output_seqlen,
                stride_dq_seqlen,
                stride_dk_seqlen,
                stride_dv_seqlen,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=True,  # use atomic operations for parallel writes
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                BLOCK_Q_ROWS_SIZE=BLOCK_Q_ROWS_SIZE,
                BLOCK_KV_COLS_SIZE=BLOCK_KV_COLS_SIZE,
            )
        else:
            # Sequential mode: Loop over key-value column blocks
            num_kv_cols_blocks = tl.cdiv(seqlen_k, BLOCK_KV_COLS_SIZE)
            for start_kv_cols_block in range(0, num_kv_cols_blocks):
                _backward_kernel_one_col_block(
                    start_kv_cols_block,
                    Q,
                    K,
                    V,
                    DOutput,
                    DQ,
                    DK,
                    DV,
                    LSE,
                    Delta,
                    softmax_scale,
                    stride_q_seqlen,
                    stride_k_seqlen,
                    stride_v_seqlen,
                    stride_d_output_seqlen,
                    stride_dq_seqlen,
                    stride_dk_seqlen,
                    stride_dv_seqlen,
                    seqlen_q,
                    seqlen_k,
                    headdim,
                    ATOMIC_ADD=False,
                    BLOCK_HEADDIM=BLOCK_HEADDIM,
                    BLOCK_Q_ROWS_SIZE=BLOCK_Q_ROWS_SIZE,
                    BLOCK_KV_COLS_SIZE=BLOCK_KV_COLS_SIZE,
                )

    _BACKWARD_KERNEL_CACHE[config_key] = _create_backward_kernel

    return _create_backward_kernel


@triton.jit
def _backward_preprocess_do_o_dot(
    Output,
    DOutput,
    Delta,
    stride_output_batch,
    stride_output_head,
    stride_output_seqlen,
    stride_d_output_batch,
    stride_d_output_head,
    stride_d_output_seqlen,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_Q_ROWS_SIZE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """Triton JIT-compiled kernel for backward pass gradient preprocessing.

    Compute initial gradient deltas for attention output gradients using
    element-wise product of output activations and output gradients.

    Args:
        Output: Forward pass output tensor pointer of shape (batch, nheads, seqlen_q, headdim).
        DOutput: Output gradient tensor pointer of shape (batch, nheads, seqlen_q, headdim).
        Delta: Intermediate tensor of shape (batch, nheads, seqlen_q) interpreted
            as 2D tensor (batch*nheads, seqlen_q_rounded) in a kernel.
        stride_*: Strides along corresponding dimensions.
        nheads: Number of attention heads.
        seqlen_q: Sequence length of queries.
        headdim: Dimension size of each attention head.
        BLOCK_Q_ROWS_SIZE: Block size for processing query rows (constexpr).
            Controls tiling along sequence dimension for Q.
        BLOCK_HEADDIM: Block size for head dimension (constexpr).

    Notes:
        1. First step in backward pass chain rule computation.
        2. Designed for sequence lengths divisible by BLOCK_Q_ROWS_SIZE.
    """
    # Determine Q block boundaries
    start_q_rows_block = tl.program_id(0)

    # Initialize offsets
    offset_batch_head = tl.program_id(1)
    offset_batch = offset_batch_head // nheads
    offset_head = offset_batch_head % nheads

    offsets_q_block_rows = start_q_rows_block * BLOCK_Q_ROWS_SIZE + tl.arange(0, BLOCK_Q_ROWS_SIZE)
    offsets_headdim = tl.arange(0, BLOCK_HEADDIM)

    # Load output block with mask for boundaries check
    output_block_mask = (offsets_q_block_rows[:, None] < seqlen_q) & (
        offsets_headdim[None, :] < headdim
    )
    output_block = tl.load(
        Output
        + offset_batch * stride_output_batch
        + offset_head * stride_output_head
        + (offsets_q_block_rows[:, None] * stride_output_seqlen + offsets_headdim[None, :]),
        mask=output_block_mask,
        other=0.0,
    ).to(tl.float32)
    d_output_block = tl.load(
        DOutput
        + offset_batch * stride_d_output_batch
        + offset_head * stride_d_output_head
        + offsets_q_block_rows[:, None] * stride_d_output_seqlen
        + offsets_headdim[None, :],
        mask=output_block_mask,
        other=0.0,
    ).to(tl.float32)

    # Compute per-row dot product of d_output and output block
    # (i.e. accumulate gradients for each query row) and store the result (write-back).
    delta = tl.sum(d_output_block * output_block, axis=1)
    tl.store(Delta + offset_batch_head * seqlen_q_rounded + offsets_q_block_rows, delta)
